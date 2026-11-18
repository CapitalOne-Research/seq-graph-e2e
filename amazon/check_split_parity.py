#!/usr/bin/env python3
"""Check whether the temporal-baseline node splits match the RelGT/RelGNN FraudTask arena.

Two split sources exist for the fraud task and may NOT be the same partition:
  * Temporal baselines (notebooks/temporal_graph.py): boolean node masks in
    <processed>/amazon-<ds>/splits/<ds>_splits_seed<seed>.pt, indexed by the node_mapping
    all_user_ids = sorted(unique(all_nodes.uid | all_edges.uid | all_edges.other_uid)).
  * RelGT / RelGNN (FraudTask): <tasks>/amazon-<ds>/fraud_detection/relational/fraud_{train,val,test}.csv
    (uid,timestamp,fraud_label), which derive from the canonical label_{train,val,test}.csv.

This maps the .pt masks back to uids and set-compares them against fraud_*.csv (and label_*.csv
if present): sizes, proportions, Jaccard per split, exact-partition check, and fraud rate.
Prints a compact summary only — no raw data leaves the box.

Run on a box that has both trees, e.g.:
  python check_split_parity.py \
    --processed ../data/processed \
    --tasks     ../data/tasks
"""
import argparse
import os

import pandas as pd
import torch


def temporal_split_uids(processed, ds, seed):
    base = os.path.join(processed, f"amazon-{ds}")
    pt = os.path.join(base, "splits", f"{ds}_splits_seed{seed}.pt")
    nodes_p, edges_p = os.path.join(base, "all_nodes.csv"), os.path.join(base, "all_edges.csv")
    if not (os.path.exists(pt) and os.path.exists(nodes_p) and os.path.exists(edges_p)):
        return None
    nodes_df, edges_df = pd.read_csv(nodes_p), pd.read_csv(edges_p)
    all_user_ids = sorted(pd.concat([nodes_df["uid"], edges_df["uid"], edges_df["other_uid"]]).unique())
    num_users = len(all_user_ids)
    s = torch.load(pt, map_location="cpu")
    out = {}
    for split in ["train", "val", "test"]:
        mask = s[f"{split}_mask"].bool()
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        out[split] = {all_user_ids[i] for i in idx if i < num_users}
    return out


def arena_split_uids(tasks, ds, which):
    """which='fraud' -> relational/fraud_*.csv ; which='label' -> fraud_detection/label_*.csv"""
    base = os.path.join(tasks, f"amazon-{ds}", "fraud_detection")
    sub = "relational" if which == "fraud" else ""
    prefix = "fraud_" if which == "fraud" else "label_"
    out, rate = {}, {}
    for split in ["train", "val", "test"]:
        p = os.path.join(base, sub, f"{prefix}{split}.csv")
        if not os.path.exists(p):
            return None, None
        df = pd.read_csv(p)
        out[split] = set(df["uid"].tolist())
        rate[split] = round(float(df["fraud_label"].mean()), 4) if "fraud_label" in df else None
    return out, rate


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def compare(name_a, A, name_b, B):
    if A is None or B is None:
        print(f"    [skip {name_a} vs {name_b}: one side missing]")
        return
    identical = all(A[s] == B[s] for s in ["train", "val", "test"])
    print(f"    {name_a} vs {name_b}:  {'IDENTICAL partition' if identical else 'DIFFER'}")
    ua, ub = set().union(*A.values()), set().union(*B.values())
    print(f"      universe: {name_a}={len(ua)} uids, {name_b}={len(ub)} uids, "
          f"overlap={len(ua & ub)} (Jaccard {jaccard(ua, ub):.3f})")
    for s in ["train", "val", "test"]:
        print(f"      {s:<5} {name_a}={len(A[s]):>6} {name_b}={len(B[s]):>6}  Jaccard={jaccard(A[s], B[s]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed", help="root holding amazon-<ds>/all_nodes.csv + splits/")
    ap.add_argument("--tasks", default="data/tasks", help="root holding amazon-<ds>/fraud_detection/")
    ap.add_argument("--datasets", default="clothing,electronics,musical,movies")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    for ds in args.datasets.split(","):
        print(f"\n=== amazon-{ds} ===")
        fraud, fraud_rate = arena_split_uids(args.tasks, ds, "fraud")
        label, label_rate = arena_split_uids(args.tasks, ds, "label")
        if fraud_rate:
            print(f"  RelGT arena fraud rate (train/val/test): "
                  f"{fraud_rate['train']}/{fraud_rate['val']}/{fraud_rate['test']}")
        if label:
            compare("label_csv", label, "fraud_csv(relgt)", fraud)
        for seed in args.seeds.split(","):
            tmp = temporal_split_uids(args.processed, ds, int(seed))
            if tmp is None:
                print(f"  seed {seed}: [no temporal .pt / all_nodes.csv found]")
                continue
            szs = {s: len(tmp[s]) for s in ["train", "val", "test"]}
            tot = sum(szs.values()) or 1
            print(f"  seed {seed}: temporal sizes {szs} "
                  f"(props {szs['train']/tot:.2f}/{szs['val']/tot:.2f}/{szs['test']/tot:.2f})")
            compare(f"temporal_s{seed}", tmp, "fraud_csv(relgt)", fraud)


if __name__ == "__main__":
    main()
