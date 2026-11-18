"""Assemble a wall-clock cost table for the fraud methods from the training logs.

Every fraud trainer logs `train_wall_sec` (LocalLogger runs also log `wall_sec_total`):
  - static_graph_clf.py          -> logs_canonical/static-graph-fraud-<ds>/*.json   (LocalLogger list)
  - review_embedding.py          -> logs_canonical/fraud-<ds>-rating/*.json          (LocalLogger list)
      (bridge / seq_only / supervised-frozen; run_name encodes which)
  - temporal_graph*.py           -> logs*/temporal-graph-fraud-<ds>/*.json           (LocalLogger list)
  - RelGT (main_node_ddp.py)     -> ../relgt-main/results/fraud_default_canon/amazon-<ds>/fraud/*.json (dict)

Reports mean +/- std TRAIN MINUTES per (method, dataset) over seeds and writes a LaTeX table.
Run from notebooks/ (same CWD as get_latex_tables.py):
    python collect_cost_table.py
    python collect_cost_table.py --relgt_root ../relgt-main/results/fraud_default_canon --out tables/fraud_cost.tex

NOTE: wall-clock is only comparable across methods run on the same GPU class; state the hardware in
the caption. Method labels mirror the results table; tweak the *_label helpers if any run is mislabeled.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

DATASETS = ["movies", "electronics", "clothing"]
_MODEL = {"transformerconv": "TFConv", "gcn": "GCN", "gat": "GAT"}


def _localjson_wall_sec(path):
    """LocalLogger format = list of dicts; prefer train_wall_sec, fall back to wall_sec_total."""
    try:
        payload = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, list):
        return None
    for key in ("train_wall_sec", "wall_sec_total"):
        vals = [e[key] for e in payload if isinstance(e, dict) and e.get(key) is not None]
        if vals:
            return float(vals[-1])
    return None


def _relgt_wall_sec(path):
    """RelGT results JSON = dict with train_wall_sec."""
    try:
        obj = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(obj, dict) and obj.get("train_wall_sec") is not None:
        return float(obj["train_wall_sec"])
    return None


def _static_label(fname):
    model = _MODEL.get(os.path.basename(fname).split("_")[0].lower(), os.path.basename(fname).split("_")[0])
    return f"{model} (Two-Stage)" if "_sbert" in os.path.basename(fname) else f"{model} (Static)"


def _rating_label(fname):
    base = os.path.basename(fname)
    if base.startswith("seq_only"):
        return "SBert"
    frozen = base.startswith("supfrozen")
    body = base.split("bridge_", 1)[1] if "bridge_" in base else base
    layer0 = body.split("-", 1)[0].split("_", 1)[0].lower()          # 'gcn-gcn...' -> 'gcn'
    model = _MODEL.get(layer0, layer0.upper())
    return f"Bridge-{model}" + (" (sup-frozen)" if frozen else "")


def _temporal_label(fname):
    return os.path.basename(fname).split("_")[0]                     # TGN / DyRep / ...


def _collect(sources):
    # sources: list of (label_fn, wall_fn, glob_pattern) -> {(method, ds): [minutes...]}
    bucket = defaultdict(list)
    missing = []
    for ds in DATASETS:
        for label_fn, wall_fn, pat in sources:
            for p in glob.glob(pat.format(ds=ds)):
                sec = wall_fn(p)
                if sec is None:
                    missing.append(p)
                    continue
                bucket[(label_fn(p), ds)].append(sec / 60.0)
    return bucket, missing


def _fmt(vals):
    if not vals:
        return "--"
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return rf"${m:.1f}\,{{\scriptstyle \pm {s:.1f}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relgt_root", default="../relgt-main/results/fraud_default_canon")
    ap.add_argument("--relgt_root_text", default="../relgt-main/results/fraud_default_canon_text")
    ap.add_argument("--out", default="tables/fraud_cost.tex")
    args = ap.parse_args()

    sources = [
        (_static_label, _localjson_wall_sec, "logs_canonical/static-graph-fraud-{ds}/*.json"),
        (_rating_label, _localjson_wall_sec, "logs_canonical/fraud-{ds}-rating/*.json"),
        (_temporal_label, _localjson_wall_sec, "logs/temporal-graph-fraud-{ds}/*.json"),
        (_temporal_label, _localjson_wall_sec, "logs_tuned_rev/temporal-graph-fraud-{ds}/*.json"),
        (_temporal_label, _localjson_wall_sec, "logs_tuned-2/temporal-graph-fraud-{ds}/*.json"),
        (lambda p: "RelGT", _relgt_wall_sec, args.relgt_root + "/amazon-{ds}/fraud/*.json"),
        (lambda p: "RelGT (+text)", _relgt_wall_sec, args.relgt_root_text + "/amazon-{ds}/fraud/*.json"),
    ]
    bucket, missing = _collect(sources)

    methods = sorted({m for (m, _) in bucket})
    # console summary
    print(f"\n{'Method':<26} " + " ".join(f"{ds:>16}" for ds in DATASETS))
    for m in methods:
        cells = []
        for ds in DATASETS:
            v = bucket.get((m, ds), [])
            cells.append(f"{mean(v):.1f}±{(stdev(v) if len(v) > 1 else 0.0):.1f} (n{len(v)})" if v else "--")
        print(f"{m:<26} " + " ".join(f"{c:>16}" for c in cells))
    if missing:
        print(f"\n[warn] {len(missing)} log(s) had no wall-clock field; first few:")
        for p in missing[:5]:
            print("   ", p)

    # LaTeX
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Training wall-clock cost (minutes, mean$\pm$std over seeds) on a single GPU. "
        r"Report the GPU class used; times are comparable only within the same hardware.}",
        r"\label{tab:fraud_cost}",
        r"\begin{tabular}{l" + "c" * len(DATASETS) + "}", r"\toprule",
        r"\textbf{Method} & " + " & ".join(rf"\textbf{{{ds.capitalize()}}}" for ds in DATASETS) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        row = (m if m.startswith("Bridge") else m)  # labels already plain
        lines.append(row + " & " + " & ".join(_fmt(bucket.get((m, ds), [])) for ds in DATASETS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(tex)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
