"""Generate the 3-dataset fraud LaTeX table INCLUDING an RDL block (RelGT / RelGNN).

Reuses the existing generator (`log_table_utils`) for the static / text / BRIDGE rows exactly as
the shared table utilities do, then appends a "Relational Deep Learning" section built from the
RelGT/RelGNN result JSONs.

RDL caveats baked in:
  * relbench metrics = average_precision (PR AUC) + roc_auc only -> no Max F1, so Max F1 cells render
    as '--'. (PR AUC is stored as a FRACTION 0-1; the generator's formatter multiplies by 100.)
  * RelGT/RelGNN never ran on Movies -> those cells render as '--'.
  * val-selected test metrics (the trainers select the best-val checkpoint).

Run from the directory containing the experiment logs (normally notebooks/):
    python get_latex_tables.py
    python get_latex_tables.py --rdl_root_full ../relgt-main/results/fraud_default ...
"""
import argparse
import glob
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

import log_table_utils as L


def _rdl_rows(dataset, roots):
    """One aggregated row per RDL method present for `dataset`: PR AUC (fraction) mean/std, Max F1 = NaN."""
    rows = []
    for label, root in roots.items():
        aps, f1s = [], []
        for p in glob.glob(os.path.join(root, "**", f"amazon-{dataset}", "fraud", "*.json"), recursive=True):
            try:
                obj = json.load(open(p))
            except (json.JSONDecodeError, OSError):
                continue
            tm = obj.get("test_metrics") if isinstance(obj, dict) else None
            if tm:
                pr = tm.get("pr_auc", tm.get("average_precision"))  # prefer trapezoidal auc(recall,precision)
                if pr is not None:
                    aps.append(float(pr))                           # fraction 0-1; generator x100s it
                    if tm.get("max_f1") is not None:
                        f1s.append(float(tm["max_f1"]))
        if aps:
            f1_mean = mean(f1s) if f1s else np.nan
            f1_std = (stdev(f1s) if len(f1s) > 1 else 0.0) if f1s else np.nan
            rows.append({"Method": label, "Max F1": f1_mean, "Max F1_std": f1_std,
                         "PR AUC": mean(aps), "PR AUC_std": (stdev(aps) if len(aps) > 1 else 0.0)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    # canonical (per-seed-split) result dirs; text-fed RelGT lives in its OWN dir, never mixed w/ no-text
    ap.add_argument("--rdl_root_full", default="../relgt-main/results/fraud_default_canon")
    ap.add_argument("--rdl_root_matched", default="../relgt-main/results/fraud_canon")
    ap.add_argument("--rdl_root_full_text", default="../relgt-main/results/fraud_default_canon_text")
    ap.add_argument("--rdl_root_relgnn", default="../RelGNN-main/examples/results/relgnn_fraud")
    ap.add_argument("--out", default="tables/fraud_three_with_rdl.tex")
    args = ap.parse_args()

    rdl_roots = {
        "RelGT (full)": args.rdl_root_full,               # categorical / no text
        "RelGT (matched)": args.rdl_root_matched,
        "RelGT (full, +text)": args.rdl_root_full_text,   # same pooled SBert as static+SBert
        "RelGNN": args.rdl_root_relgnn,
    }
    for _label, _root in rdl_roots.items():
        if not glob.glob(os.path.join(_root, "**", "*.json"), recursive=True):
            print(f"[warn] RDL root for '{_label}' has no JSONs: {_root}  (row will render as --)")

    ds_key = {"movies": "Amazon-Movies", "electronics": "Amazon-Electronics", "clothing": "Amazon-Clothing"}
    dfs = {}
    for ds, key in ds_key.items():
        base, _ = L.process_dataset_directly(ds)          # static/text/BRIDGE rows (existing pipeline)
        rdl = _rdl_rows(ds, rdl_roots)
        dfs[key] = pd.concat([base, rdl], ignore_index=True) if not rdl.empty else base

    # existing default sections; \ourmodel (Bridge) group ALWAYS rendered last (bottom of the table).
    sections = (
        ("Graph Only (Static)", ["GCN", "GAT", "TransformerConv"]),
        ("Temporal Graphs", ["TGN", "DyRep"]),
        ("Two-Stage", []),
        ("Sequence Only (Review Text Embedding)", ["SBert", "SBert Seq"]),
        ("Relational Deep Learning (RDL, categorical / no text)",
         ["RelGT (full)", "RelGT (matched)", "RelGNN"]),
        ("Relational Deep Learning (+ review text)", ["RelGT (full, +text)"]),
        (r"\ourmodel", ["SBert + GCN", "SBert + GAT", "SBert + TransformerConv", "SBert + TokenXAT",
                        r"\ourmodel-GCN", r"\ourmodel-GAT", r"\ourmodel-TransformerConv", r"\ourmodel-TokenXAT"]),
    )

    tex = L.generate_three_dataset_table(dfs, sections=sections)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(tex)
    print(tex)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
