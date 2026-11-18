#!/usr/bin/env python3
"""Degree-stratified analysis of test predictions (the "for whom" figure).

Consumes the per-edge prediction dumps produced by `joint.py --load_checkpoint_and_eval <best.pth>
--dump_predictions` (files `<run>_test_preds.npz` with arrays src, dst, rank). Bins each test
positive by the QUERY user's (src) train-graph degree and reports MRR / Hits@k per bin with support
counts and bootstrap 95% CIs, for each model. This is how we test whether the end-to-end gain
concentrates on low-degree / cold-start users.

Compare the per-bin MRR here to the co-location floor per bin from `colocation_leakage.py` — a gain
is only meaningful if it clears the floor in that bin.

Usage:
  python degree_stratified.py --task_dir ../data/tasks/gowalla/friend_recommendation \
      --dumps bridge=path/to/bridge_test_preds.npz \
              unfrozen=path/to/unfrozen_test_preds.npz \
              frozen=path/to/frozen_test_preds.npz

No training. pandas + numpy.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

DEGREE_BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 10**9)]


def bin_label(lo, hi):
    if lo == hi:
        return f"{lo}"
    return f">{lo-1}" if hi >= 10**9 else f"{lo}-{hi}"


def train_degree(task_dir):
    tr = pd.read_csv(os.path.join(task_dir, "relational_train.csv"), usecols=["uid", "other_uid"])
    deg = {}
    for a, b in zip(tr["uid"].to_numpy(), tr["other_uid"].to_numpy()):
        if pd.notna(a):
            deg[int(a)] = deg.get(int(a), 0) + 1
        if pd.notna(b):
            deg[int(b)] = deg.get(int(b), 0) + 1
    return deg


def bootstrap_ci(values, stat_fn, n_boot=1000, alpha=0.05, seed=0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = stat_fn(values[idx])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def analyze(dump_path, deg, k_list, n_boot):
    d = np.load(dump_path)
    src, rank = d["src"], d["rank"].astype(np.float64)
    src_deg = np.array([deg.get(int(u), 0) for u in src])
    rows = []
    for lo, hi in DEGREE_BINS:
        m = (src_deg >= lo) & (src_deg <= hi)
        n = int(m.sum())
        if n == 0:
            rows.append((bin_label(lo, hi), 0, float("nan"), (float("nan"), float("nan")), {}))
            continue
        rr = 1.0 / rank[m]
        mrr = float(rr.mean())
        ci = bootstrap_ci(rr, np.mean, n_boot=n_boot)
        hits = {k: float((rank[m] <= k).mean()) for k in k_list}
        rows.append((bin_label(lo, hi), n, mrr, ci, hits))
    overall_mrr = float((1.0 / rank).mean())
    return rows, overall_mrr, len(rank)


def main():
    ap = argparse.ArgumentParser(description="Degree-stratified analysis of test-prediction dumps")
    ap.add_argument("--task_dir", required=True, help="Dir with relational_train.csv (for degree)")
    ap.add_argument("--dumps", nargs="+", required=True, help="label=path.npz entries (one per model)")
    ap.add_argument("--hits_k", default="1,10")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    k_list = [int(x) for x in args.hits_k.split(",")]

    deg = train_degree(args.task_dir)
    models = []
    for entry in args.dumps:
        if "=" not in entry:
            sys.exit(f"--dumps entries must be label=path (got '{entry}')")
        label, path = entry.split("=", 1)
        if not os.path.exists(path):
            sys.exit(f"dump not found: {path}")
        models.append((label, analyze(path, deg, k_list, args.n_boot)))

    # Per-model overall
    print("=== Overall MRR ===")
    for label, (_, overall, n) in models:
        print(f"  {label:>16}: MRR={overall:.4f}  (n={n})")

    # Per-bin table (MRR with 95% CI)
    print("\n=== MRR by query-user train-degree bin  [mean (95% CI), n] ===")
    header = "  degree | " + " | ".join(f"{label:^26}" for label, _ in models)
    print(header)
    bins = [bin_label(lo, hi) for lo, hi in DEGREE_BINS]
    # rows aligned across models by bin index
    for bi, blabel in enumerate(bins):
        cells = []
        for _, (rows, _, _) in models:
            _, n, mrr, (lo, hi), _ = rows[bi]
            if n == 0:
                cells.append(f"{'--':^26}")
            else:
                cells.append(f"{mrr:.3f} [{lo:.3f},{hi:.3f}] n={n:<5}".center(26))
        print(f"  {blabel:>6} | " + " | ".join(cells))

    print("\nInterpretation: if the end-to-end model's per-bin MRR rises above the 2-stage model's")
    print("MOST at LOW degree (non-overlapping CIs), that is the 'for-whom' finding. Cross-check each")
    print("bin against the co-location floor (colocation_leakage.py) — a gain must clear that floor.")


if __name__ == "__main__":
    main()
