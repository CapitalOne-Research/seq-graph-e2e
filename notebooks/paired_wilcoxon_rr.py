#!/usr/bin/env python3
"""Paired per-query significance for the link / ranking task (MRR).

MRR = mean of per-query reciprocal rank (RR), so MRR *decomposes per query* and the
right test is paired over queries: for two models evaluated on the SAME test queries,
test whether the per-query RR differences are centered at 0 (Wilcoxon signed-rank) and
bootstrap the MRR gap for a CI. Thousands of queries -> well-powered, unlike the 3-seed
test. This is the link-task counterpart of paired_bootstrap_prauc.py; use one convention
across both tasks so significance is applied uniformly (no cherry-picking).

Inputs: one .npz per model per seed, produced by your link eval at the best-val
checkpoint (see the dump snippet in chat):
    np.savez(path, qid=<int[]>, rr=<float[]>)        # rr = 1 / rank of the true relation
Both models' files for a seed must cover the same queries (aligned here by qid).

Usage:
    python paired_wilcoxon_rr.py \
        --a 'preds_link/bridge_brightkite_seed*.npz'     --name-a "BRIDGE (E)" \
        --b 'preds_link/supfrozenD_brightkite_seed*.npz'  --name-b "sup-frozen (D)"
"""
import argparse
import glob
import re
import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:                                   # scipy optional; bootstrap still works
    wilcoxon = None


def _seed_of(p):
    m = re.search(r"seed(\d+)", p)
    return int(m.group(1)) if m else -1


def _load(p):
    d = np.load(p)
    files = getattr(d, "files", list(d.keys()))
    if "rr" in files and "qid" in files:                 # generic (qid, rr) format
        return d["qid"].astype(str), d["rr"].astype(np.float64)
    # joint.py link dump (test(dump_path=...)): src, dst, rank  ->  rr = 1/rank,
    # query id = the (src,dst) positive test edge (unique + stable across models on a seed).
    src, dst, rank = d["src"], d["dst"], d["rank"].astype(np.float64)
    qid = np.array([f"{int(s)}_{int(t)}" for s, t in zip(src, dst)])
    return qid, 1.0 / rank


def align(pa, pb):
    qa, ra = _load(pa)
    qb, rb = _load(pb)
    a = dict(zip(qa, ra))
    b = dict(zip(qb, rb))
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(f"no shared qids between {pa} and {pb}")
    return np.array([a[q] for q in common]), np.array([b[q] for q in common])


def boot_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = diff.size
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = diff[rng.integers(0, n, n)].mean()
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="glob for model-A per-seed npz")
    ap.add_argument("--b", required=True, help="glob for model-B per-seed npz")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    fa = {_seed_of(p): p for p in glob.glob(args.a)}
    fb = {_seed_of(p): p for p in glob.glob(args.b)}
    seeds = sorted(set(fa) & set(fb))
    if not seeds:
        print(f"MISSING matching seeds. A={sorted(fa)} B={sorted(fb)}")
        return

    print(f"=== MRR: ({args.name_a}) - ({args.name_b})  [paired over queries] ===")
    gaps = []
    for s in seeds:
        ra, rb = align(fa[s], fb[s])
        diff = ra - rb
        gap = float(diff.mean())
        gaps.append(gap)
        lo, hi = boot_ci(diff, args.n_boot, s)
        p = float("nan")
        if wilcoxon is not None and np.any(diff != 0):
            try:
                _, p = wilcoxon(ra, rb, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                p = float("nan")
        sig = "" if lo <= 0 <= hi else "  *SIG*"
        print(f" seed {s}: MRR {args.name_a} {ra.mean():.4f}  {args.name_b} {rb.mean():.4f}"
              f"  gap {gap:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  wilcoxon p={p:.2e}"
              f"  (n_q={ra.size}){sig}")
    g = np.array(gaps)
    std = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f" across seeds: mean MRR gap {g.mean():+.4f} +/- {std:.4f}")
    print(" *SIG* = that seed's 95% bootstrap CI excludes 0. Report TIE unless CIs exclude 0 across seeds.")


if __name__ == "__main__":
    main()
