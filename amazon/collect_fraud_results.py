#!/usr/bin/env python3
"""Collect fraud-detection results (BRIDGE, temporal/static baselines, RelGT) into one table.

Scans result dirs for per-run JSONs, extracts test ROC-AUC / PR-AUC, and aggregates
mean +/- std across seeds, grouped by (dataset, model, sweep-dir, hparam-signature) so
different sweeps/hparams never get mixed. Prints the trivial-feature floor as a reference.

Two JSON schemas are handled:
  * BRIDGE / temporal / static baselines: a LIST of per-epoch dicts; the final entries
    carry test_roc_auc / test_pr_auc (+ best_test_*).
  * RelGT (main_node_ddp.py): {"val_metrics": {...}, "test_metrics": {...}}.

Run on a box that can see the result dirs, e.g.:
  python collect_fraud_results.py \
      ~/event-seq-graph-icml-old/notebooks ~/event-seq-graph-non-shared/notebooks \
      --relgt ~/event-seq-graph-icml/relgt-main/results \
      --source logs_tuned-2          # optional: only the tuned BRIDGE sweep

Positional args = any roots holding fraud-*/temporal-graph-fraud-*/static-graph-fraud-* dirs.
--relgt = RelGT results root (the --out_dir you trained into); scans */amazon-<ds>/fraud/<seed>.json.
"""
import argparse
import glob
import gzip
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, stdev

# longest names first so substring matching is unambiguous
DATASETS = ["home-kitchen", "electronics", "clothing", "musical", "movies"]
TRIVIAL_FLOOR = {"clothing": 0.742, "musical": 0.773}  # AUC; others not computed
TEMPORAL = ["DyGFormer", "GraphMixer", "DyRep", "CTAN", "TPNet", "TGN", "NAT", "TCL"]


def load_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def dataset_of(path):
    p = path.lower()
    for ds in DATASETS:
        if ds in p:
            return ds
    return None


def seed_of(name):
    m = re.search(r"seed_?(\d+)", name)
    return int(m.group(1)) if m else None


def model_of(fname):
    n = fname.lower()
    if n.startswith("relgnn"):
        return "RelGNN"
    if n.startswith("relgt"):
        return "RelGT"
    if "tokenxat" in n:
        return "TokenXAT"
    if n.startswith(("bridge_gcn-gcn", "gcn-gcn")):
        return "BRIDGE-GCN"
    if n.startswith(("bridge_gat-gat", "gat-gat")):
        return "BRIDGE-GAT"
    if n.startswith("seq_only"):
        return "Seq-only"
    for m in TEMPORAL:
        if n.startswith(m.lower()):
            return m
    if n.startswith("gatv2"):
        return "GATv2 (static)"
    if n.startswith("gat"):
        return "GAT (static)"
    return None


def sig_of(fname):
    """hparam signature = filename minus seed + extension (so seeds aggregate cleanly)."""
    s = re.sub(r"\.json(\.gz)?$", "", fname)
    s = re.sub(r"_?seed_?\d+", "", s)
    return s


def sweep_of(path):
    """the logs_* sweep dir + immediate parent, to keep sweeps and old/new runs separate."""
    parts = path.split(os.sep)
    logs = next((p for p in parts if p.startswith("logs")), None)
    parent = os.path.basename(os.path.dirname(path))
    return f"{logs}/{parent}" if logs else parent


def extract(obj):
    """-> dict(auc, pr, best_auc, best_pr, selected) from either schema, or None."""
    if isinstance(obj, dict) and "test_metrics" in obj:  # RelGT
        tm = obj.get("test_metrics") or {}
        return {"auc": tm.get("roc_auc"), "pr": tm.get("average_precision"),
                "best_auc": None, "best_pr": None, "selected": "val-selected"}
    if isinstance(obj, list):  # BRIDGE / baseline per-epoch list
        tests = [e for e in obj if isinstance(e, dict) and "test_roc_auc" in e]
        if not tests:
            return None
        last = tests[-1]
        return {"auc": last.get("test_roc_auc"), "pr": last.get("test_pr_auc"),
                "best_auc": last.get("best_test_roc_auc"), "best_pr": last.get("best_test_pr_auc"),
                "selected": "final-epoch"}
    if isinstance(obj, dict) and "test_roc_auc" in obj:
        return {"auc": obj.get("test_roc_auc"), "pr": obj.get("test_pr_auc"),
                "best_auc": obj.get("best_test_roc_auc"), "best_pr": obj.get("best_test_pr_auc"),
                "selected": "final-epoch"}
    return None


def add(rows, key, seed, m):
    rows[key].append((seed, m))


def agg(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def fmt(a):
    return f"{a[0]:.4f}+-{a[1]:.3f}" if a else "    -     "


def main():
    ap = argparse.ArgumentParser(description="Collect fraud results into one table")
    ap.add_argument("roots", nargs="*", help="dirs to scan for baseline/BRIDGE fraud JSONs")
    ap.add_argument("--relgt", action="append", default=[],
                    help="RelGT results root(s); scans **/amazon-<ds>/fraud/<seed>.json")
    ap.add_argument("--source", default=None,
                    help="only include groups whose sweep dir contains this substring")
    ap.add_argument("--best", action="store_true",
                    help="emit per-(dataset,model) best-config markdown, dropping degenerate runs")
    args = ap.parse_args()

    # key = (dataset, model, sweep, sig) -> list of (seed, metrics)
    rows = defaultdict(list)
    n_files = n_parsed = 0

    # 1) generic baseline / BRIDGE result JSONs
    for root in args.roots:
        for path in glob.glob(os.path.join(root, "**", "*.json*"), recursive=True):
            if not re.search(r"fraud", path, re.I):
                continue
            fname = os.path.basename(path)
            if fname == "best_hparams.json":
                continue
            n_files += 1
            ds, model, seed = dataset_of(path), model_of(fname), seed_of(fname)
            if ds is None or model is None or seed is None:
                continue
            try:
                m = extract(load_json(path))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] parse fail {path}: {e}")
                continue
            if not m:
                continue
            add(rows, (ds, model, sweep_of(path), sig_of(fname)), seed, m)
            n_parsed += 1

    # 2) RelGT results: <root>/**/amazon-<ds>/fraud/<seed>.json
    for root in args.relgt:
        for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            if "amazon-" not in path or (os.sep + "fraud") not in path:
                continue
            n_files += 1
            ds = dataset_of(path)
            b = os.path.basename(path)
            seed = seed_of(b)
            if seed is None:
                mm = re.match(r"(\d+)\.json$", b)
                seed = int(mm.group(1)) if mm else None
            if ds is None or seed is None:
                continue
            try:
                m = extract(load_json(path))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] parse fail {path}: {e}")
                continue
            if not m:
                continue
            pre = path.split("amazon-")[0].rstrip(os.sep)
            sweep = os.path.basename(pre) if pre else "relgt"  # e.g. fraud_default vs fraud
            add(rows, (ds, "RelGT", sweep, "relgt"), seed, m)
            n_parsed += 1

    # best-config-per-model markdown (degenerate runs dropped)
    if args.best:
        best = {}
        for key in rows:
            ds, model = key[0], key[1]
            recs = [m for _, m in rows[key]]
            n = len({s for s, _ in rows[key]})
            auc = agg([m["auc"] for m in recs])
            if auc is None:
                continue
            if n >= 2 and auc[1] == 0.0:  # identical across seeds -> degenerate/broken
                continue
            pr = agg([m["pr"] for m in recs])
            cur = best.get((ds, model))
            if cur is None or auc[0] > cur["auc"][0]:
                best[(ds, model)] = {"auc": auc, "pr": pr, "n": n}
        print("# Fraud comparison — best config per model\n")
        print("_Per (dataset, model): max mean test-AUC; degenerate runs (identical across "
              "seeds) dropped. Test-selected is optimistic — for the final paper table use "
              "val-selected models from the tuned sweep._\n")
        for ds in sorted({k[0] for k in best}):
            floor = TRIVIAL_FLOOR.get(ds)
            print("\n## " + ds + (f"  (trivial floor AUC={floor})" if floor else ""))
            print("| model | test AUC | test PR-AUC | n | vs floor |")
            print("|---|---|---|---|---|")
            for (d, model) in sorted(k for k in best if k[0] == ds):
                b = best[(d, model)]
                a, p = b["auc"], b["pr"]
                prs = f"{p[0]:.4f} ± {p[1]:.3f}" if p else "—"
                vf = ("above" if (floor and a[0] >= floor) else "below") if floor else "—"
                print(f"| {model} | {a[0]:.4f} ± {a[1]:.3f} | {prs} | {b['n']} | {vf} |")
        return

    # aggregate + print
    print(f"\nParsed {n_parsed}/{n_files} candidate files.\n")
    hdr = (f"{'dataset':<10} {'model':<14} {'sweep':<30} {'cfg':<28} {'n':>2}  "
           f"{'test AUC':>13}  {'test PR-AUC':>13}  {'best AUC':>12}")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(rows):
        ds, model, sweep, sig = key
        if args.source and args.source not in sweep:
            continue
        recs = [m for _, m in rows[key]]
        seeds = sorted({s for s, _ in rows[key]})
        auc = agg([m["auc"] for m in recs])
        pr = agg([m["pr"] for m in recs])
        bauc = agg([m["best_auc"] for m in recs])
        floor = TRIVIAL_FLOOR.get(ds)
        flag = ""
        if auc and floor is not None:
            flag = "  ^floor" if auc[0] >= floor else "  v<floor"
        print(f"{ds:<10} {model:<14} {sweep[-30:]:<30} {sig[-28:]:<28} {len(seeds):>2}  "
              f"{fmt(auc):>13}  {fmt(pr):>13}  {fmt(bauc):>12}{flag}")

    print("\nTrivial-feature floors (AUC):  " +
          "   ".join(f"{d}={v:.3f}" for d, v in TRIVIAL_FLOOR.items()))
    print("Columns: test AUC = last-epoch test (val-safe-ish); best AUC = best_test_* "
          "(may be test-peeked -- confirm the selection protocol before reporting).")


if __name__ == "__main__":
    main()
