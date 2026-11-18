#!/usr/bin/env python3
"""Single source of truth for the Amazon fraud-detection train/val/test split.

Why this module exists
----------------------
Historically each fraud trainer invented its own split:
  - static_graph_clf.py / review_embedding.py (BRIDGE): user-only graph, appearance order
    (`nodes_df['uid'].unique()`), RandomNodeSplit default 80/10/10.
  - temporal_graph.py / temporal_graph_tpnet.py: users+event nodes, `sorted` order,
    RandomNodeSplit default 60/20/20.
  - the arena (task_amazon-*.ipynb) + trivial floor: sklearn 70/10/20.
Splits were cached per (dataset, seed) at splits/<ds>_splits_seed<seed>.pt, and whichever
trainer ran first "won" — with different node sets and orderings the caches were not even
interpretable across families. The reported table therefore mixed several different splits.

This module defines ONE canonical, user-level split that every consumer shares. The canonical
split is anchored to the split the (spared) temporal/BRIDGE-era runs used: it is recovered from
temporal's cached RandomNodeSplit .pt (60/20/20, users+events, users-first in `sorted` order),
materialized to a plain, inspectable CSV, and then mapped into any model's own node indexing via
`masks_for_uids`. RelGT/RelGNN and the trivial floor build their arena from the same CSV.
"""
import os

import numpy as np
import pandas as pd
import torch


def split_dir_for(base_path):
    return os.path.join(base_path, "splits")


def canonical_split_path(base_path, dataset, seed):
    return os.path.join(split_dir_for(base_path), f"{dataset}_canonical_user_split_seed{seed}.csv")


def _temporal_user_order(base_path):
    """Reproduce temporal_graph.py's user node ordering exactly (users are nodes [0, num_users))."""
    nodes = pd.read_csv(os.path.join(base_path, "all_nodes.csv"), usecols=["uid"])
    edges = pd.read_csv(os.path.join(base_path, "all_edges.csv"), usecols=["uid", "other_uid"])
    return sorted(pd.concat([nodes["uid"], edges["uid"], edges["other_uid"]]).unique())


def _recover_from_temporal_pt(base_path, dataset, seed):
    """Recover {uid: split} from temporal's cached RandomNodeSplit .pt, or None if absent."""
    pt = os.path.join(split_dir_for(base_path), f"{dataset}_splits_seed{seed}.pt")
    if not os.path.exists(pt):
        return None
    users = _temporal_user_order(base_path)
    n_users = len(users)
    users = np.asarray(users, dtype=object)
    sp = torch.load(pt, map_location="cpu", weights_only=False)
    out = {}
    for name, lab in (("train_mask", "train"), ("val_mask", "val"), ("test_mask", "test")):
        m = sp[name].bool().cpu().numpy().ravel()
        if len(m) < n_users:
            raise ValueError(f"{pt}: {name} length {len(m)} < n_users {n_users} — split/nodes mismatch")
        for u in users[m[:n_users]]:
            out[u] = lab
    return out


def _fresh_user_split(base_path, seed, val=0.2, test=0.2):
    """Fresh user-level split (canonical fraction 60/20/20) over labeled users; only used when no
    temporal .pt exists for this (dataset, seed)."""
    users = pd.unique(pd.read_csv(os.path.join(base_path, "all_nodes.csv"), usecols=["uid"])["uid"])
    n = len(users)
    perm = np.random.RandomState(seed).permutation(n)
    n_test, n_val = int(round(test * n)), int(round(val * n))
    out = {}
    for i in perm[:n_test]:
        out[users[i]] = "test"
    for i in perm[n_test:n_test + n_val]:
        out[users[i]] = "val"
    for i in perm[n_test + n_val:]:
        out[users[i]] = "train"
    return out


def get_canonical_user_split(base_path, dataset, seed):
    """Return {uid: 'train'|'val'|'test'} — the one split every fraud method must use.

    Precedence: cached canonical CSV -> recovered from temporal's .pt -> fresh 60/20/20.
    The result is written to splits/<ds>_canonical_user_split_seed<seed>.csv so it is stable,
    inspectable, and identical across methods and runs.
    """
    csv = canonical_split_path(base_path, dataset, seed)
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        return dict(zip(df["uid"], df["split"]))

    split = _recover_from_temporal_pt(base_path, dataset, seed)
    source = "temporal_pt"
    if split is None:
        split = _fresh_user_split(base_path, seed)
        source = "fresh_60_20_20"

    os.makedirs(split_dir_for(base_path), exist_ok=True)
    df = pd.DataFrame({"uid": list(split.keys()), "split": list(split.values())}).sort_values("uid")
    tmp = f"{csv}.tmp.{os.getpid()}"  # atomic write: concurrent workers can't read a half-written CSV
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv)
    n = len(df)
    tr, va, te = ((df["split"] == s).sum() for s in ("train", "val", "test"))
    print(f"[fraud_splits] {dataset} seed{seed}: source={source} n={n} "
          f"train/val/test={tr}/{va}/{te} ({tr/n:.2f}/{va/n:.2f}/{te/n:.2f}) -> {csv}")
    return split


def masks_for_uids(uid_split, uid_map, num_nodes):
    """Build boolean train/val/test masks aligned to a model's own node indexing.

    uid_split: {uid: split}  (canonical, user-level)
    uid_map:   {uid: node_index} for this model (dict or pd.Series); users not in the graph and
               non-user nodes (events) are simply left False (they are excluded via y != -1).
    """
    if hasattr(uid_map, "to_dict"):
        uid_map = uid_map.to_dict()
    buckets = {"train": [], "val": [], "test": []}
    for uid, s in uid_split.items():
        j = uid_map.get(uid)
        if j is not None and 0 <= j < num_nodes:
            buckets[s].append(j)
    masks = []
    for s in ("train", "val", "test"):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        if buckets[s]:
            m[torch.tensor(buckets[s], dtype=torch.long)] = True
        masks.append(m)
    return masks[0], masks[1], masks[2]
