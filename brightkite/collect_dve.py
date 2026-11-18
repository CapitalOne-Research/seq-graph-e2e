#!/usr/bin/env python3
"""Collect the D-vs-E link control from joint.py result JSONs.

Reads {dataset}_results_seeded/*.json, classifies each run by its filename tag
  E  = end-to-end baseline (NO control tag; produced by launch_exp.py --config baseline)
  D  = supervised-frozen        (_supfrozen_D)
  i  = unsupervised-frozen      (_frozenseq_matched)   [MLM-pretrained, frozen]
  u  = unfrozen two-stage       (_unfrozen2stage)      [MLM-pretrained, fine-tuned jointly]
extracts the best-val-selected test metrics (best_test_mrr / best_test_hits_at_k,
falling back to test_*), and prints mean+/-std over seeds per variant plus the E-D gap.

Only runs matching --match (default the reported config `gcn-gcn2x2`, nngb10) and NOT
matching --exclude (default `bs128`, the old sweep) are counted, so the stale bs128 sweep
and other backbones are ignored. Run from brightkite/:
    python collect_dve.py --datasets brightkite gowalla

NOTE: this gives the aggregate MRR/Hits +/-std (the D-vs-E *numbers*). The paired-per-query
significance is separate -- dump per-query RR from joint.py and use paired_wilcoxon_rr.py.
"""
import argparse
import glob
import json
import os
import re
from statistics import mean, stdev

# (filename tag suffix, display label) -- order = display order
TAGS = [
    (None,                "E (end-to-end)"),
    ("supfrozen_D",       "D (sup-frozen)"),
    ("frozenseq_matched", "i (unsup-frozen)"),
    ("unfrozen2stage",    "u (unfrozen 2-stg)"),
]
LABELS = [lab for _, lab in TAGS]


def classify(basename):
    for tag, label in TAGS:
        if tag and basename.endswith(f"_{tag}.json"):
            return label
    return "E (end-to-end)"          # no control tag -> baseline


def seed_of(basename):
    m = re.search(r"_seed(\d+)", basename)
    return int(m.group(1)) if m else None


def best_test(path, k_list):
    try:
        data = json.load(open(path))
    except Exception:
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    for prefix in ("best_test_", "test_"):     # prefer the val-selected checkpoint
        row = next((r for r in reversed(data)
                    if isinstance(r, dict) and f"{prefix}mrr" in r), None)
        if row:
            out = {"mrr": float(row[f"{prefix}mrr"])}
            for k in k_list:
                v = row.get(f"{prefix}hits_at_{k}")
                if v is not None:
                    out[f"hits@{k}"] = float(v)
            return out
    return None


def _pct(v):                                   # JSON stores fractions in [0,1]; show as %
    return v * 100 if v is not None and v <= 1.5 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["brightkite", "gowalla"])
    ap.add_argument("--match", default="gcn-gcn2x2", help="substring a run name must contain")
    ap.add_argument("--exclude", default="bs128", help="substring to exclude (old config)")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5])
    args = ap.parse_args()

    for ds in args.datasets:
        print(f"\n===== {ds}  (match='{args.match}', exclude='{args.exclude}') =====")
        bucket = {}                            # label -> {seed: metrics}
        for p in glob.glob(f"{ds}_results_seeded/*.json"):
            b = os.path.basename(p)
            if args.match not in b or args.exclude in b:
                continue
            if any(j in b for j in ("smoke", "debug", "_epochs2_", "_epochs1_", "_epochs5_")):
                continue  # skip smoke/debug/short-epoch runs
            s = seed_of(b)
            if s is None:
                continue
            m = best_test(p, args.k)
            if m is None:
                continue
            bucket.setdefault(classify(b), {})[s] = m

        cols = ["variant", "n", "MRR"] + [f"H@{k}" for k in args.k]
        print("  " + "".join(f"{c:>18}" for c in cols))
        agg_mrr = {}
        for lab in LABELS:
            runs = bucket.get(lab, {})
            if not runs:
                print("  " + f"{lab:>18}" + f"{'--':>18}")
                continue

            def ms(key):
                vs = [r[key] for r in runs.values() if key in r]
                return (mean(vs), stdev(vs) if len(vs) > 1 else 0.0) if vs else None

            agg_mrr[lab] = ms("mrr")
            cells = [f"{lab:>18}", f"{len(runs):>18}"]
            for key in ["mrr"] + [f"hits@{k}" for k in args.k]:
                r = ms(key)
                cells.append(f"{_pct(r[0]):.1f}±{_pct(r[1]):.1f}" if r else "--")
            print("  " + "".join(f"{c:>18}" for c in cells))

        e, d = agg_mrr.get("E (end-to-end)"), agg_mrr.get("D (sup-frozen)")
        if e and d:
            print(f"  --> E - D (MRR): {_pct(e[0]) - _pct(d[0]):+.2f} pts  "
                  f"(E={_pct(e[0]):.1f}, D={_pct(d[0]):.1f}); seeds E={len(bucket['E (end-to-end)'])}, D={len(bucket['D (sup-frozen)'])}")
        elif d and not e:
            print("  [!] D present but E (end-to-end baseline) NOT found at this config "
                  "-- run launch_exp.py --config baseline at the SAME bs/backbone, or the D-vs-E is bs-mismatched.")
        elif e and not d:
            print("  [!] E present but D (supfrozen_D) not finished yet.")


if __name__ == "__main__":
    main()
