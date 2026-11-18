#!/usr/bin/env python3
"""Build a normalized relational schema (tables + PK-FK + time) for the Amazon fraud task,
so RelGT / RelGNN (RelBench-native) and BRIDGE can all be run on the SAME data/split/metric.

This is the shared substrate for the fair RDL comparison. It emits the dataframes a relbench
`Database` needs; a thin relbench `Dataset`/`EntityTask` wrapper (built on top, validated on the
box) turns these into what `make_pkey_fkey_graph` consumes.

Input (per amazon-<ds>/fraud_detection/):
  events.csv         : uid,timestamp,event_set,event(="ASIN:rating" | "co-review_product"),other_uid
  label_{train,val,test}.csv : uid,fraud_label
Output (-> <task_dir>/relational/):
  users.csv      pkey=uid
  products.csv   pkey=asin
  reviews.csv    pkey=review_id, fkey uid->users, asin->products, time=timestamp, feat=rating
  coreview.csv   pkey=edge_id,  fkey uid->users, other_uid->users, time=timestamp   (user-user relation)
  fraud_{train,val,test}.csv : uid(entity, fkey->users), timestamp(seed time), fraud_label(target)
  schema.json    : declares pkey / fkeys / time_col per table (drives the relbench Database)

Fairness: every model sees the SAME tables. No table carries a feature derived from the label.
Report all models against the trivial-feature floor from fraud_trivial_baseline.py.

pandas only; validated locally.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd


def build(task_dir, out_dir):
    ev = pd.read_csv(os.path.join(task_dir, "events.csv"))
    per = ev[ev["event_set"] == "personal"].copy()
    rel = ev[ev["event_set"] == "relational"].copy()

    # reviews (personal events): event = "ASIN:rating"
    parts = per["event"].astype(str).str.split(":", n=1, expand=True)
    per["asin"] = parts[0]
    per["rating"] = pd.to_numeric(parts[1], errors="coerce")
    reviews = per[["uid", "asin", "timestamp", "rating"]].reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(len(reviews)))

    # entities
    users = pd.DataFrame({"uid": pd.unique(pd.concat([
        ev["uid"], ev["other_uid"].dropna()]).astype("int64"))}).sort_values("uid").reset_index(drop=True)
    products = pd.DataFrame({"asin": pd.unique(reviews["asin"])}).reset_index(drop=True)

    # user-user co-review relation (two FKs to users)
    coreview = rel[["uid", "other_uid", "timestamp"]].dropna(subset=["other_uid"]).copy()
    coreview["other_uid"] = coreview["other_uid"].astype("int64")
    coreview = coreview.reset_index(drop=True)
    coreview.insert(0, "edge_id", np.arange(len(coreview)))

    # task tables: static fraud label per user; seed time = user's last review (all own history visible)
    last_ts = reviews.groupby("uid")["timestamp"].max()
    global_ts = int(reviews["timestamp"].max())
    task_tables = {}
    for split in ["train", "val", "test"]:
        lab = pd.read_csv(os.path.join(task_dir, f"label_{split}.csv"))
        lab["timestamp"] = lab["uid"].map(last_ts).fillna(global_ts).astype("int64")
        task_tables[split] = lab[["uid", "timestamp", "fraud_label"]]

    os.makedirs(out_dir, exist_ok=True)
    users.to_csv(f"{out_dir}/users.csv", index=False)
    products.to_csv(f"{out_dir}/products.csv", index=False)
    reviews.to_csv(f"{out_dir}/reviews.csv", index=False)
    coreview.to_csv(f"{out_dir}/coreview.csv", index=False)
    for split, t in task_tables.items():
        t.to_csv(f"{out_dir}/fraud_{split}.csv", index=False)

    schema = {
        "tables": {
            "users":    {"pkey": "uid",  "fkeys": {}, "time_col": None},
            "products": {"pkey": "asin", "fkeys": {}, "time_col": None},
            "reviews":  {"pkey": "review_id",
                          "fkeys": {"uid": "users", "asin": "products"},
                          "time_col": "timestamp"},
            "coreview": {"pkey": "edge_id",
                          "fkeys": {"uid": "users", "other_uid": "users"},
                          "time_col": "timestamp"},
        },
        "task": {"name": "fraud", "type": "binary_classification",
                  "entity_table": "users", "entity_col": "uid",
                  "time_col": "timestamp", "target_col": "fraud_label",
                  "metrics": ["roc_auc", "average_precision"],
                  "tables": {s: f"fraud_{s}.csv" for s in ["train", "val", "test"]}},
    }
    with open(f"{out_dir}/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # ---- sanity / FK integrity ----
    uset = set(users["uid"])
    checks = {
        "users": len(users), "products": len(products), "reviews": len(reviews),
        "coreview_edges": len(coreview),
        "reviews.uid in users": bool(set(reviews["uid"]).issubset(uset)),
        "reviews.asin in products": bool(set(reviews["asin"]).issubset(set(products["asin"]))),
        "coreview.uid in users": bool(set(coreview["uid"]).issubset(uset)),
        "coreview.other_uid in users": bool(set(coreview["other_uid"]).issubset(uset)),
        "fraud_train/val/test rows": {s: len(t) for s, t in task_tables.items()},
        "fraud rate (test)": round(float(task_tables["test"]["fraud_label"].mean()), 4),
        "task users in users table": bool(
            set(pd.concat([task_tables[s]["uid"] for s in task_tables])).issubset(uset)),
    }
    return checks


def main():
    ap = argparse.ArgumentParser(description="Build relational schema for the fraud RDL comparison")
    ap.add_argument("--data_root", default="../data/tasks")
    ap.add_argument("--datasets", nargs="+", default=["clothing", "electronics", "movies"])
    ap.add_argument("--task", default="fraud_detection",
                    help="task subdir, e.g. fraud_detection_seed0 for the per-seed canonical arena")
    args = ap.parse_args()
    any_ok = False
    for ds in args.datasets:
        task_dir = os.path.join(args.data_root, f"amazon-{ds}", args.task)
        if not os.path.isdir(task_dir):
            print(f"[skip] {ds}: {task_dir} not found")
            continue
        any_ok = True
        out_dir = os.path.join(task_dir, "relational")
        checks = build(task_dir, out_dir)
        print(f"\n=== amazon-{ds} -> {out_dir} ===")
        for k, v in checks.items():
            print(f"  {k}: {v}")
    if not any_ok:
        sys.exit("No fraud task dirs found.")


if __name__ == "__main__":
    main()
