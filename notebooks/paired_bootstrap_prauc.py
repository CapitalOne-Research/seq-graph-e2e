#!/usr/bin/env python3
"""Paired bootstrap over users: significance of the PR-AUC / Max-F1 gap between
two fraud models, evaluated on the SAME test users (paired), per seed.

Method
------
Each test user has (predicted P(fraud), true label). Two models are evaluated on
the SAME users, so we PAIR by user: resample users with replacement (the SAME
indices for BOTH models), recompute the *set-level* metric for each model on the
resample, and take the gap. 10k resamples -> 95% CI + two-sided bootstrap p for
H0: gap = 0. The metric (auc(recall,precision) / max-F1) is a property of the
whole ranking, so it is NOT averaged per user -- it is recomputed on each
resampled set. Stratified resampling (positives / negatives separately) keeps
prevalence fixed, which matters on the low-prevalence categories.

Inputs: one .npz per model per seed, produced by your eval at the best-val
checkpoint (see dump snippet in the chat):
    np.savez(path, uid=<int[]>, score=<float[]>, label=<0/1 int[]>)
Both models' files for a seed must cover the same test users (aligned here by uid).

Usage
-----
    python paired_bootstrap_prauc.py \
        --a 'preds/relgt_clothing_seed*.npz'  --name-a "RelGT+text" \
        --b 'preds/bridge-gat_clothing_seed*.npz' --name-b "BRIDGE-GAT" \
        --metric both --n-boot 10000
"""
import argparse
import glob
import re
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def pr_auc(label, score):
    p, r, _ = precision_recall_curve(label, score)
    return auc(r, p)                       # auc(recall, precision) == harmonized metric


def max_f1(label, score):
    p, r, _ = precision_recall_curve(label, score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(f1.max())


METRICS = {"pr_auc": pr_auc, "max_f1": max_f1}


def _seed_of(path):
    m = re.search(r"seed(\d+)", path)
    return int(m.group(1)) if m else -1


def _load(path):
    d = np.load(path)
    return d["uid"].astype(np.int64), d["score"].astype(np.float64), d["label"].astype(np.int64)


def align(pa, pb):
    """Return (y, score_a, score_b) over the users shared by both files, in uid order."""
    ua, sa, la = _load(pa)
    ub, sb, lb = _load(pb)
    a = {u: (s, l) for u, s, l in zip(ua, sa, la)}
    b = {u: (s, l) for u, s, l in zip(ub, sb, lb)}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(f"no shared uids between {pa} and {pb}")
    y = np.array([a[u][1] for u in common])
    if not np.array_equal(y, np.array([b[u][1] for u in common])):
        raise ValueError(f"labels disagree across {pa} and {pb} (misaligned users?)")
    sa2 = np.array([a[u][0] for u in common])
    sb2 = np.array([b[u][0] for u in common])
    return y, sa2, sb2


def paired_bootstrap(y, sa, sb, metric, n_boot=10000, stratified=True, seed=0):
    rng = np.random.default_rng(seed)
    f = METRICS[metric]
    obs = f(y, sa) - f(y, sb)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n = y.size
    d = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(pos, pos.size, True),
                                  rng.choice(neg, neg.size, True)])
        else:
            idx = rng.integers(0, n, n)
        yy = y[idx]
        s = yy.sum()
        if s == 0 or s == yy.size:          # degenerate resample -> skip
            continue
        d[k] = f(yy, sa[idx]) - f(yy, sb[idx])
        k += 1
    d = d[:k]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return dict(obs=obs, lo=lo, hi=hi, p=p, ma=f(y, sa), mb=f(y, sb),
                n=n, n_pos=int(y.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="glob for model-A per-seed npz")
    ap.add_argument("--b", required=True, help="glob for model-B per-seed npz")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--metric", choices=list(METRICS) + ["both"], default="both")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--no-stratified", action="store_true")
    args = ap.parse_args()

    metrics = list(METRICS) if args.metric == "both" else [args.metric]
    fa = {_seed_of(p): p for p in glob.glob(args.a)}
    fb = {_seed_of(p): p for p in glob.glob(args.b)}
    seeds = sorted(set(fa) & set(fb))
    if not seeds:
        print(f"MISSING matching seeds. A={sorted(fa)} B={sorted(fb)}")
        return

    for metric in metrics:
        print(f"\n=== {metric}: ({args.name_a}) - ({args.name_b})  [paired bootstrap over users] ===")
        gaps = []
        for s in seeds:
            y, sa, sb = align(fa[s], fb[s])
            r = paired_bootstrap(y, sa, sb, metric, args.n_boot,
                                 not args.no_stratified, seed=s)
            gaps.append(r["obs"])
            sig = "" if r["lo"] <= 0 <= r["hi"] else "  *SIG*"
            print(f" seed {s}: {args.name_a} {100*r['ma']:5.1f}  {args.name_b} {100*r['mb']:5.1f}"
                  f"  gap {100*r['obs']:+5.2f}  95% CI [{100*r['lo']:+5.2f}, {100*r['hi']:+5.2f}]"
                  f"  p={r['p']:.3f}  (n={r['n']}, n_pos={r['n_pos']}){sig}")
        g = np.array(gaps)
        std = g.std(ddof=1) if len(g) > 1 else 0.0
        allsig = "TIE" if (g.min() <= 0 <= g.max() or len(g) < 2) else "consistent-sign"
        print(f" across seeds: mean gap {100*g.mean():+.2f} +/- {100*std:.2f}   [{allsig}]")
        print(" *SIG* = that seed's 95% CI excludes 0. Report TIE unless CIs exclude 0 across all seeds.")


if __name__ == "__main__":
    main()
