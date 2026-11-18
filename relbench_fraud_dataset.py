#!/usr/bin/env python3
"""Custom RelBench Dataset + EntityTask for our Amazon fraud data, so RelGT and RelGNN
(both RelBench-native, node/entity tasks) run on the SAME data/split/metric as BRIDGE.

Built from the tables emitted by amazon/build_relational_tables.py (users/products/reviews/
coreview + fraud_{train,val,test}.csv). Design (grounded in relbench base classes read locally):
  * Dense-remap uid/asin to 0..N-1 integers (relbench.validate_and_correct_db requires
    consecutive pkeys; we do it up front so reindex is identity and the task tables stay aligned).
  * Force asin/id columns to categorical at the RelGT/RelGNN call site (no text embedder ->
    dodges the blocked HuggingFace/GloVe download).
  * Our split is BY USER, not by time, so override get_table() to return our fixed splits
    directly (bypasses relbench's temporal make_table machinery).

Self-test on the box:  python relbench_fraud_dataset.py ../data/tasks/amazon-clothing/fraud_detection/relational
"""
import os
import sys
import numpy as np
import pandas as pd

# Use the retained core for the adapter's standalone self-check as well.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "RelGNN-main"))

from relbench.base import Database, Dataset, Table, EntityTask
from relbench.base.task_base import TaskType
from relbench.metrics import average_precision, roc_auc


class FraudDataset(Dataset):
    def __init__(self, rel_dir: str, cache_dir=None, use_review_text=False):
        self.rel_dir = rel_dir
        self.use_review_text = use_review_text
        users = pd.read_csv(os.path.join(rel_dir, "users.csv"))
        products = pd.read_csv(os.path.join(rel_dir, "products.csv"))
        # dense integer maps (relbench needs consecutive pkeys; keep task tables aligned)
        self.uid2idx = {u: i for i, u in enumerate(sorted(users["uid"].tolist()))}
        self.asin2idx = {a: i for i, a in enumerate(products["asin"].tolist())}
        # Per-user pooled SBert (row = dense uid), used as a review-text feature to match static+SBert.
        self.user_sbert = None
        if use_review_text:
            p = os.path.join(rel_dir, "user_sbert.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"user_sbert.npy not found in {rel_dir}; build it with amazon/build_user_sbert.py")
            self.user_sbert = np.load(p)
        rv = pd.read_csv(os.path.join(rel_dir, "reviews.csv"), usecols=["timestamp"])
        max_ts = pd.to_datetime(rv["timestamp"].max(), unit="s")
        # our split is by user, not time; set timestamps so upto() keeps ALL rows + __init__ passes
        self.test_timestamp = max_ts
        self.val_timestamp = max_ts - pd.Timedelta(days=1)
        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        d = self.rel_dir
        users = pd.read_csv(f"{d}/users.csv")
        products = pd.read_csv(f"{d}/products.csv")
        reviews = pd.read_csv(f"{d}/reviews.csv")
        coreview = pd.read_csv(f"{d}/coreview.csv")

        users["uid"] = users["uid"].map(self.uid2idx)
        products["asin"] = products["asin"].map(self.asin2idx)
        if self.use_review_text:
            # key that the lookup text-embedder in main_node_ddp maps back to user_sbert[row]
            users["sbert_key"] = users["uid"].astype(str)
        reviews["uid"] = reviews["uid"].map(self.uid2idx)
        reviews["asin"] = reviews["asin"].map(self.asin2idx)
        reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="s")
        coreview["uid"] = coreview["uid"].map(self.uid2idx)
        coreview["other_uid"] = coreview["other_uid"].map(self.uid2idx)
        coreview["timestamp"] = pd.to_datetime(coreview["timestamp"], unit="s")

        return Database(table_dict={
            "users": Table(df=users, fkey_col_to_pkey_table={}, pkey_col="uid"),
            "products": Table(df=products, fkey_col_to_pkey_table={}, pkey_col="asin"),
            "reviews": Table(df=reviews,
                             fkey_col_to_pkey_table={"uid": "users", "asin": "products"},
                             pkey_col="review_id",
                             time_col="timestamp"),
            "coreview": Table(df=coreview,
                              fkey_col_to_pkey_table={"uid": "users", "other_uid": "users"},
                              pkey_col="edge_id",
                              time_col="timestamp"),
        })


class FraudTask(EntityTask):
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "uid"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "fraud_label"
    timedelta = pd.Timedelta(days=1)
    metrics = [average_precision, roc_auc]
    num_eval_timestamps = 1

    def get_table(self, split, mask_input_cols=None):
        # Override: return our fixed per-user split directly (no temporal make_table).
        if mask_input_cols is None:
            mask_input_cols = split == "test"
        df = pd.read_csv(os.path.join(self.dataset.rel_dir, f"fraud_{split}.csv"))
        df["uid"] = df["uid"].map(self.dataset.uid2idx)
        df = df.dropna(subset=["uid"])
        df["uid"] = df["uid"].astype("int64")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        cols = ["uid", "timestamp"] + ([] if mask_input_cols else ["fraud_label"])
        return Table(df=df[cols],
                     fkey_col_to_pkey_table={"uid": "users"},
                     pkey_col=None, time_col="timestamp")


if __name__ == "__main__":
    rel = sys.argv[1] if len(sys.argv) > 1 else "../data/tasks/amazon-clothing/fraud_detection/relational"
    ds = FraudDataset(rel)
    db = ds.get_db()
    print("DB tables:", {k: len(t.df) for k, t in db.table_dict.items()})
    task = FraudTask(ds)
    for split in ["train", "val", "test"]:
        t = task.get_table(split)
        print(f"{split}: {len(t.df)} rows, cols={list(t.df.columns)}")
    print("OK — FraudDataset + FraudTask build. Feed these to RelGT/RelGNN instead of get_dataset/get_task.")
