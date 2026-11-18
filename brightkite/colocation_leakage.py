#!/usr/bin/env python3
"""Co-location leakage audit for the friendship (relationship) task on check-in datasets.

Analog of amazon/leakage_diagnostic.py, adapted for Brightkite/Gowalla. Unlike Amazon
(where the edge-defining product IDs were literally tokenized into inputs), friendship
edges come from a SEPARATE social graph and friend IDs never appear in the sequences.
The residual risk is CO-LOCATION HOMOPHILY: friends check in at the same places, so a
zero-training counter of shared check-in geohashes between two users' sequences may rival
trained models -- especially at LOW degree, which is exactly where the degree-inversion
claim (joint task-adaptation helps sparsely-connected users) lives.

We score each candidate pair by the number of shared check-in geohashes (truncated to a
given precision, mirroring the model's geohash prefix sub-tokens), rank each positive
against its negatives with the exact eval protocol (positive at index 0, mid-rank for ties
so the score cannot exploit favorable tie-breaking), and report the trivial MRR/Hits@k
OVERALL and STRATIFIED BY TRAIN-GRAPH DEGREE.

Read: if this trivial baseline approaches the trained models -- particularly in the low
degree bins -- the "joint adaptation helps low-degree users" finding is confounded by
co-location and the friendship task cannot license it.

No training, no torch. Pandas/numpy only. Runs on either conda env.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

DATASETS = ["brightkite", "gowalla"]
DEGREE_BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 10**9)]


def load_locations(personal_csv, precision):
    """uid -> set of check-in geohashes truncated to `precision` chars."""
    df = pd.read_csv(personal_csv, usecols=["uid", "event_set", "event"])
    if "event_set" in df.columns:
        df = df[df["event_set"] == "personal"]
    ev = df["event"].astype(str).str.slice(0, precision)
    locs = {}
    for uid, gh in zip(df["uid"].to_numpy(), ev.to_numpy()):
        locs.setdefault(int(uid), set()).add(gh)
    return locs


def train_degree(task_dir):
    """Undirected train-graph degree per uid, from relational_train.csv."""
    tr = pd.read_csv(os.path.join(task_dir, "relational_train.csv"), usecols=["uid", "other_uid"])
    deg = {}
    for a, b in zip(tr["uid"].to_numpy(), tr["other_uid"].to_numpy()):
        if pd.notna(a):
            deg[int(a)] = deg.get(int(a), 0) + 1
        if pd.notna(b):
            deg[int(b)] = deg.get(int(b), 0) + 1
    return deg


def pair_scores(uids, other_uids, locs):
    out = np.zeros(len(uids), dtype=np.float64)
    for i, (u, v) in enumerate(zip(uids, other_uids)):
        su = locs.get(int(u)) if pd.notna(u) else None
        sv = locs.get(int(v)) if pd.notna(v) else None
        out[i] = 0.0 if (not su or not sv) else float(len(su & sv))
    return out


def midrank(pos_scores, neg_scores, k_list):
    """Returns (rank[N], metrics dict). Positive at index 0; ties split (mid-rank)."""
    gt = (neg_scores > pos_scores[:, None]).sum(axis=1)
    eq = (neg_scores == pos_scores[:, None]).sum(axis=1)
    rank = 1.0 + gt + 0.5 * eq
    m = {"MRR": float((1.0 / rank).mean())}
    for k in k_list:
        m[f"Hits@{k}"] = float((rank <= k).mean())
    return rank, m


def evaluate(task_dir, split, k_list, neg_suffix, precision):
    locs = load_locations(os.path.join(task_dir, "personal.csv"), precision)
    pos = pd.read_csv(os.path.join(task_dir, f"relational_{split}.csv"))
    neg = pd.read_csv(os.path.join(task_dir, f"relational_{split}{neg_suffix}.csv"))
    n_pos = len(pos)
    if n_pos == 0 or len(neg) % n_pos != 0:
        raise RuntimeError(f"Neg ({len(neg)}) not a multiple of pos ({n_pos}); cannot align.")
    K = len(neg) // n_pos

    pos_s = pair_scores(pos["uid"].to_numpy(), pos["other_uid"].to_numpy(), locs)
    neg_s = pair_scores(neg["uid"].to_numpy(), neg["other_uid"].to_numpy(), locs).reshape(n_pos, K)
    aligned = bool(np.all(neg["uid"].to_numpy().reshape(n_pos, K) == pos["uid"].to_numpy()[:, None]))

    rank, overall = midrank(pos_s, neg_s, k_list)

    # degree stratification by the query user's train degree
    deg = train_degree(task_dir)
    pos_deg = np.array([deg.get(int(u), 0) for u in pos["uid"].to_numpy()])
    strata = []
    for lo, hi in DEGREE_BINS:
        mask = (pos_deg >= lo) & (pos_deg <= hi)
        n = int(mask.sum())
        mrr = float((1.0 / rank[mask]).mean()) if n else float("nan")
        strata.append((lo, hi, n, mrr))

    frac_shared = float((pos_s > 0).mean())
    return overall, strata, dict(n_pos=n_pos, K=K, aligned=aligned,
                                 frac_shared=frac_shared,
                                 mean_pos=float(pos_s.mean()), mean_neg=float(neg_s.mean()))


def main():
    ap = argparse.ArgumentParser(description="Co-location leakage audit (friendship task)")
    ap.add_argument("--data_root", default="../data/tasks")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--task", default="friend_recommendation")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--neg_suffix", default="_negative_sample_100",
                    help="_negative_sample_100 (100 negs) or _negative_sample (full).")
    ap.add_argument("--precisions", default="4,6,8",
                    help="Geohash char precisions to test (coarse->fine).")
    ap.add_argument("--hits_k", default="1,5,10,50,100")
    args = ap.parse_args()
    k_list = [int(x) for x in args.hits_k.split(",")]
    precisions = [int(x) for x in args.precisions.split(",")]

    any_ok = False
    for ds in args.datasets:
        task_dir = os.path.join(args.data_root, ds, args.task)
        if not os.path.isdir(task_dir):
            print(f"[skip] {ds}: {task_dir} not found")
            continue
        any_ok = True
        print(f"\n===== {ds}  ({args.split})  ->  {task_dir} =====")
        for p in precisions:
            overall, strata, d = evaluate(task_dir, args.split, k_list, args.neg_suffix, p)
            geo = {4: "~20km", 5: "~2.4km", 6: "~1.2km", 7: "~150m", 8: "~19m"}.get(p, "")
            print(f"\n  --- geohash precision {p} ({geo}) | pos={d['n_pos']} negs/pos={d['K']} aligned={d['aligned']} ---")
            print(f"  trivial co-location MRR={overall['MRR']:.4f}  " +
                  "  ".join(f"H@{k}={overall[f'Hits@{k}']:.3f}" for k in k_list))
            print(f"  pos share >=1 location: {d['frac_shared']*100:.1f}%   mean overlap pos={d['mean_pos']:.2f} neg={d['mean_neg']:.2f}")
            print("  by train-degree bin (query user):")
            for lo, hi, n, mrr in strata:
                label = f"{lo}" if lo == hi else (f">{lo-1}" if hi >= 10**9 else f"{lo}-{hi}")
                print(f"     deg {label:>6}: n={n:>6}  trivial MRR={mrr:.4f}")

    if not any_ok:
        sys.exit("No task dirs found. Check --data_root / --datasets / --task.")
    print("\nInterpretation: compare each degree bin's trivial MRR to the TRAINED models' MRR in")
    print("that bin. If trivial rivals trained at LOW degree, the degree-inversion finding is")
    print("confounded by co-location homophily and must be reframed. Finer precision = stronger")
    print("(more specific) co-location signal; the model sees geohash prefix sub-tokens at all scales.")


if __name__ == "__main__":
    main()
