#!/usr/bin/env python3
"""Build a fraud arena (events.csv + label_{train,val,test}.csv) from the ONE canonical user split
that every fraud method now shares (see notebooks/fraud_splits.py).

The canonical split is anchored to the (spared) temporal/BRIDGE-era runs: it is recovered from
temporal's cached RandomNodeSplit .pt at <processed>/splits/<ds>_splits_seed<seed>.pt. This makes
RelGT/RelGNN and the trivial floor use byte-identical user membership to the neural baselines.

  python amazon/splits_pt_to_arena.py --dataset movies --seed 0 \
      --processed data/processed/amazon-movies \
      --out data/tasks/amazon-movies/fraud_detection

Run once per seed (0,1,2) with a per-seed --out to get seed-matched arenas.
"""
import argparse
import os
import sys

import pandas as pd

EVENT_COLS = ["uid", "timestamp", "event_set", "event", "other_uid"]
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "notebooks"))
from fraud_splits import get_canonical_user_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build the fraud arena from the canonical user split")
    ap.add_argument("--dataset", required=True, help="short name, e.g. movies / electronics / clothing")
    ap.add_argument("--processed", required=True, help="dir with all_nodes.csv, all_edges.csv, splits/")
    ap.add_argument("--seed", type=int, default=0, help="which canonical split seed to use (0/1/2)")
    ap.add_argument("--out", default=None,
                    help="dest fraud_detection dir (default: data/tasks/amazon-<ds>/fraud_detection)")
    args = ap.parse_args()

    ds = args.dataset
    out = args.out or os.path.join("data", "tasks", f"amazon-{ds}", "fraud_detection")
    nodes_path = os.path.join(args.processed, "all_nodes.csv")
    edges_path = os.path.join(args.processed, "all_edges.csv")
    for p in (nodes_path, edges_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")

    uid_split = get_canonical_user_split(args.processed, ds, args.seed)
    split_df = pd.DataFrame({"uid": list(uid_split.keys()), "split": list(uid_split.values())})

    nodes = pd.read_csv(nodes_path)
    if "other_uid" not in nodes.columns:
        nodes["other_uid"] = pd.NA
    edges = pd.read_csv(edges_path)

    # label_*.csv: labeled users only (those in all_nodes.csv), partitioned by the canonical split
    lab = nodes[["uid", "fraud_label"]].drop_duplicates("uid").merge(split_df, on="uid", how="inner")
    os.makedirs(out, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        sub = lab[lab["split"] == split][["uid", "fraud_label"]].sort_values("uid").reset_index(drop=True)
        sub.to_csv(os.path.join(out, f"label_{split}.csv"), index=False)
        counts[split] = (len(sub), int(sub["fraud_label"].sum()), float(sub["fraud_label"].mean()))

    ev = pd.concat([nodes[EVENT_COLS], edges[EVENT_COLS]], ignore_index=True)
    ev.to_csv(os.path.join(out, "events.csv"), index=False)

    print(f"canonical seed{args.seed}: labeled users {len(lab)} / split entries {len(split_df)}")
    for s in ("train", "val", "test"):
        n, f, r = counts[s]
        print(f"  {s}: {n} users  fraud {f}  rate {r:.4f}")
    print(f"events.csv: {len(ev)} rows ({len(nodes)} personal + {len(edges)} relational)")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
