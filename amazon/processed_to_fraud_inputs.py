#!/usr/bin/env python3
"""Convert baseline 'processed/<dataset>' data (all_nodes.csv + all_edges.csv, with a per-row
split_mask) into the RelGT/RelGNN fraud-task inputs (events.csv + label_{train,val,test}.csv) that
build_relational_tables.py consumes.

Used to add a dataset (e.g. movies) to the RDL arena using the SAME split the baselines used (the
split_mask column), so the RelGT/RelGNN cell is comparable to the BRIDGE/baseline cell.

  python amazon/processed_to_fraud_inputs.py \
      --processed ~/event-seq-graph-non-shared/data/processed/amazon-movies \
      --out data/tasks/amazon-movies/fraud_detection
"""
import argparse
import os
import pandas as pd

EVENT_COLS = ["uid", "timestamp", "event_set", "event", "other_uid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="dir with all_nodes.csv + all_edges.csv")
    ap.add_argument("--out", required=True, help="dest fraud_detection dir (events.csv + label_*.csv)")
    args = ap.parse_args()

    nodes = pd.read_csv(os.path.join(args.processed, "all_nodes.csv"))
    edges = pd.read_csv(os.path.join(args.processed, "all_edges.csv"))
    for c in ("uid", "timestamp", "event_set", "event", "fraud_label", "split_mask"):
        if c not in nodes.columns:
            raise SystemExit(f"all_nodes.csv missing column '{c}' (have {list(nodes.columns)})")
    if "other_uid" not in nodes.columns:
        nodes["other_uid"] = pd.NA

    os.makedirs(args.out, exist_ok=True)

    # events.csv = personal (nodes) + relational (edges), keeping only the 5 columns the builder reads
    ev = pd.concat([nodes[EVENT_COLS], edges[EVENT_COLS]], ignore_index=True)
    ev.to_csv(os.path.join(args.out, "events.csv"), index=False)

    # label_{split}.csv: one row per user, partitioned by the baseline split_mask
    lab = nodes[["uid", "fraud_label", "split_mask"]].drop_duplicates("uid")
    counts = {}
    for split in ("train", "val", "test"):
        sub = lab[lab["split_mask"] == split][["uid", "fraud_label"]]
        sub.to_csv(os.path.join(args.out, f"label_{split}.csv"), index=False)
        counts[split] = len(sub)

    if lab["split_mask"].nunique() < 3:
        print(f"[warn] split_mask values present: {sorted(lab['split_mask'].unique())} "
              "(expected train/val/test)")
    print(f"events.csv: {len(ev)} rows ({len(nodes)} personal + {len(edges)} relational)")
    print(f"labels train/val/test: {counts}  overall fraud rate: {lab['fraud_label'].mean():.4f}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
