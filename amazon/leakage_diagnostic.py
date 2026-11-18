#!/usr/bin/env python3
"""Leakage diagnostic for the Amazon co-review relationship task.

Tests whether the co-review target edge is recoverable directly from model INPUTS.
Co-review edges are defined by two users sharing an ASIN; the full ASIN is also
tokenized into each user's observed sequence. So a predictor that only counts
shared ASINs between two users' *observed* sequences uses nothing but the model's
inputs. If that trivial, training-free score reproduces BRIDGE's ranking metrics,
the task is leaking and cannot serve as relationship-prediction evidence.

No training, no torch. Reads the same files the model eval uses:
  <task_dir>/personal_observed.csv                     (uid, event="ASIN:rating")
  <task_dir>/relational_<split>.csv                    (uid, other_uid)  -- positives
  <task_dir>/relational_<split>_negative_sample_100.csv (uid, other_uid) -- 100 negs/positive

Ranking matches amazon/static_graph.py: candidates = [pos, 100 negs], positive at
index 0; MRR and Hits@k over the positive's mid-rank (ties split, so the score
cannot exploit favorable tie-breaking).
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

DATASETS = ["clothing", "musical", "home-kitchen", "electronics", "movies"]


def asin_of(event):
    # event looks like "B0009G1E0K:5" -> "B0009G1E0K"
    s = str(event)
    return s.split(":", 1)[0] if ":" in s else s


def load_observed_sequences(task_dir):
    """uid -> set of ASINs in that user's OBSERVED (input) sequence."""
    df = pd.read_csv(os.path.join(task_dir, "personal_observed.csv"))
    if "event_set" in df.columns:
        df = df[df["event_set"] == "personal"]
    seqs = {}
    for uid, event in zip(df["uid"].to_numpy(), df["event"].to_numpy()):
        seqs.setdefault(int(uid), set()).add(asin_of(event))
    return seqs


def pair_scores(uids, other_uids, seqs):
    out = np.empty(len(uids), dtype=np.float64)
    for i, (u, v) in enumerate(zip(uids, other_uids)):
        su = seqs.get(int(u))
        sv = seqs.get(int(v))
        out[i] = 0.0 if (not su or not sv) else float(len(su & sv))
    return out


def evaluate(task_dir, split, k_list, neg_suffix):
    seqs = load_observed_sequences(task_dir)

    pos = pd.read_csv(os.path.join(task_dir, f"relational_{split}.csv"))
    neg = pd.read_csv(os.path.join(task_dir, f"relational_{split}{neg_suffix}.csv"))
    n_pos = len(pos)
    if n_pos == 0:
        raise RuntimeError(f"No positives in relational_{split}.csv")
    if len(neg) % n_pos != 0:
        raise RuntimeError(
            f"Negatives ({len(neg)}) not a multiple of positives ({n_pos}); "
            "cannot align blocks."
        )
    K = len(neg) // n_pos

    pos_scores = pair_scores(pos["uid"].to_numpy(), pos["other_uid"].to_numpy(), seqs)
    neg_scores = pair_scores(
        neg["uid"].to_numpy(), neg["other_uid"].to_numpy(), seqs
    ).reshape(n_pos, K)

    # Alignment check: each negative block shares the positive's uid.
    neg_uid_block = neg["uid"].to_numpy().reshape(n_pos, K)
    aligned = bool(np.all(neg_uid_block == pos["uid"].to_numpy()[:, None]))

    gt = (neg_scores > pos_scores[:, None]).sum(axis=1)   # negs strictly better
    eq = (neg_scores == pos_scores[:, None]).sum(axis=1)  # ties with positive
    rank = 1.0 + gt + 0.5 * eq                            # mid-rank for ties
    rr = 1.0 / rank

    metrics = {"MRR": float(rr.mean())}
    for k in k_list:
        metrics[f"Hits@{k}"] = float((rank <= k).mean())

    # Leakage-specific signals (the honest ones to report in the paper):
    has_signal = pos_scores > 0
    diag = {
        "n_positives": int(n_pos),
        "negs_per_pos": int(K),
        "neg_blocks_aligned_to_pos_uid": aligned,
        "frac_pos_with_shared_asin": float(has_signal.mean()),
        # Unambiguous wins: positive strictly outranks every negative, no tie.
        "frac_pos_unique_top1_no_tie": float(((gt == 0) & (eq == 0)).mean()),
        # Among pairs that DO share a product, how often the positive uniquely tops.
        "frac_unique_top1_among_shared": (
            float(((gt == 0) & (eq == 0) & has_signal).sum() / max(int(has_signal.sum()), 1))
        ),
        "mean_pos_overlap": float(pos_scores.mean()),
        "mean_neg_overlap": float(neg_scores.mean()),
    }
    return metrics, diag


def main():
    ap = argparse.ArgumentParser(description="Amazon co-review leakage diagnostic")
    ap.add_argument("--data_root", default="../data/tasks",
                    help="Root containing amazon-<dataset>/coreview_prediction dirs")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--hits_k", default="1,3,5,10,20,50,100")
    ap.add_argument("--neg_suffix", default="_negative_sample_100",
                    help="Negative file suffix: _negative_sample_100 (100 negs) or "
                         "_negative_sample (full ~1000 negs, to match paper tables).")
    ap.add_argument("--task_dir", default=None,
                    help="Point directly at one coreview_prediction dir (overrides datasets)")
    args = ap.parse_args()
    k_list = [int(x) for x in args.hits_k.split(",")]

    if args.task_dir:
        targets = [(os.path.basename(os.path.dirname(args.task_dir.rstrip("/"))), args.task_dir)]
    else:
        targets = [
            (d, os.path.join(args.data_root, f"amazon-{d}", "coreview_prediction"))
            for d in args.datasets
        ]

    any_ok = False
    for name, task_dir in targets:
        if not os.path.isdir(task_dir):
            print(f"[skip] {name}: {task_dir} not found")
            continue
        any_ok = True
        metrics, diag = evaluate(task_dir, args.split, k_list, args.neg_suffix)
        print(f"\n===== {name}  ({args.split})  ->  {task_dir} =====")
        print(f"  positives={diag['n_positives']}  negs/pos={diag['negs_per_pos']}  "
              f"aligned={diag['neg_blocks_aligned_to_pos_uid']}")
        print("  --- trivial shared-ASIN predictor (INPUTS ONLY, no training) ---")
        print(f"  MRR = {metrics['MRR']:.4f}")
        print("  " + "  ".join(f"H@{k}={metrics[f'Hits@{k}']:.3f}" for k in k_list))
        print("  --- leakage signals ---")
        print(f"  positives whose two users share >=1 ASIN in input : {diag['frac_pos_with_shared_asin']*100:.1f}%")
        print(f"  positives that are the UNIQUE top-1 (no tie)        : {diag['frac_pos_unique_top1_no_tie']*100:.1f}%")
        print(f"    ... among only the shared-ASIN pairs             : {diag['frac_unique_top1_among_shared']*100:.1f}%")
        print(f"  mean overlap  pos={diag['mean_pos_overlap']:.3f}  neg={diag['mean_neg_overlap']:.3f}  "
              f"(ratio {diag['mean_pos_overlap']/max(diag['mean_neg_overlap'],1e-9):.0f}x)")

    if not any_ok:
        sys.exit("No task dirs found. Check --data_root / --task_dir.")

    print("\nInterpretation:")
    print("  Exact-ASIN overlap is a LOWER BOUND on leakage: the model also sees ASIN")
    print("  sub-tokens (asin1/2/3), so partial matches give it even more signal.")
    print("  If MRR / Hits@k here approach BRIDGE's reported numbers, the co-review task")
    print("  is largely solvable from inputs alone and must be rescoped or the product")
    print("  identity removed from the sequence tokens.")


if __name__ == "__main__":
    main()
