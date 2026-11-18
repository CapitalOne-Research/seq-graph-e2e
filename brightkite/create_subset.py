import argparse
import os
import random
from collections import defaultdict, deque
import json
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import re


def _read_required_columns(path: str, usecols: List[str], chunksize: Optional[int]):
    if chunksize and chunksize > 0:
        return pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    return [pd.read_csv(path, usecols=usecols)]


def build_undirected_adj(edge_files: List[str], chunksize: Optional[int]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for path in edge_files:
        for chunk in _read_required_columns(path, usecols=["uid", "other_uid"], chunksize=chunksize):
            uids = chunk["uid"].astype(int).to_numpy()
            vids = chunk["other_uid"].astype(int).to_numpy()
            for u, v in zip(uids, vids):
                if u == v:
                    continue
                adj[u].add(v)
                adj[v].add(u)
    return adj


def snowball_users(
    adj: Dict[int, Set[int]],
    target_size: int,
    seed_size: int,
    rng: random.Random,
) -> List[int]:
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if not adj:
        raise ValueError("No edges found to build adjacency")

    all_users = set(adj.keys())
    seed_size = min(seed_size, len(all_users), target_size)
    seeds = rng.sample(list(all_users), k=seed_size)

    selected: List[int] = []
    seen: Set[int] = set()
    q: deque = deque()

    for u in seeds:
        if u in seen:
            continue
        seen.add(u)
        selected.append(u)
        q.append(u)

    while q and len(selected) < target_size:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            selected.append(v)
            q.append(v)
            if len(selected) >= target_size:
                break

    if len(selected) < target_size:
        remaining = list(all_users - seen)
        if remaining:
            selected.extend(rng.sample(remaining, k=min(len(remaining), target_size - len(selected))))

    return selected[:target_size]


def filter_and_write_csv(
    input_path: str,
    output_path: str,
    allowed_users: Set[int],
    uid_map: Optional[Dict[int, int]],
    chunksize: Optional[int],
    is_relational: bool,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reader = pd.read_csv(input_path, chunksize=chunksize) if chunksize and chunksize > 0 else [pd.read_csv(input_path)]
    out_chunks: List[pd.DataFrame] = []

    for chunk in reader:
        if "uid" not in chunk.columns:
            raise ValueError(f"Missing 'uid' column in {input_path}")

        chunk["uid"] = chunk["uid"].astype(int)
        mask = chunk["uid"].isin(allowed_users)

        if is_relational:
            if "other_uid" not in chunk.columns:
                raise ValueError(f"Missing 'other_uid' column in {input_path}")
            chunk["other_uid"] = chunk["other_uid"].astype(int)
            mask = mask & chunk["other_uid"].isin(allowed_users)

        chunk = chunk[mask].copy()
        if chunk.empty:
            continue

        if uid_map is not None:
            chunk["uid"] = chunk["uid"].map(uid_map).astype(int)
            if is_relational:
                chunk["other_uid"] = chunk["other_uid"].map(uid_map).astype(int)

        out_chunks.append(chunk)

    if out_chunks:
        out_df = pd.concat(out_chunks, ignore_index=True)
    else:
        # preserve schema
        out_df = pd.read_csv(input_path, nrows=0)

    out_df.to_csv(output_path, index=False)


def _infer_neg_k_from_filename(filename: str, default_k: int) -> int:
    m = re.search(r"_negative_sample_(\d+)\.csv$", filename)
    if m:
        return int(m.group(1))
    return default_k


def regenerate_negative_samples(
    output_dir: str,
    allowed_users: Set[int],
    seed: int,
    default_k: int = 1000,
):
    """
    Regenerate val/test negative-sample CSVs so each positive edge has a fixed K negatives.

    This is important after subsetting, because filtering can drop negatives whose endpoints
    are outside the allowed user set, breaking the assumed "K negatives per positive" layout.
    """
    train_path = os.path.join(output_dir, "relational_train.csv")
    if not os.path.exists(train_path):
        return

    train_df = pd.read_csv(train_path)
    if train_df.empty:
        return

    # Determine which neg files exist (we only regenerate what exists).
    all_files = {name for name in os.listdir(output_dir) if name.endswith(".csv")}
    for split in ("val", "test"):
        pos_name = f"relational_{split}.csv"
        pos_path = os.path.join(output_dir, pos_name)
        if pos_name not in all_files:
            continue
        pos_df = pd.read_csv(pos_path)
        if pos_df.empty:
            continue

        # Build positives seen for this split: train + split positives, grouped by uid.
        seen = pd.concat([train_df[["uid", "other_uid"]], pos_df[["uid", "other_uid"]]], ignore_index=True)
        uid_to_pos = seen.groupby("uid")["other_uid"].agg(set).to_dict()

        # Find neg-sample filenames for this split.
        neg_names = [n for n in all_files if n.startswith(f"relational_{split}_negative_sample")]
        if not neg_names:
            continue

        rng = np.random.default_rng(seed)
        allowed_list = np.array(sorted(allowed_users), dtype=np.int64)
        allowed_py = allowed_list.tolist()

        base_cols = [c for c in pos_df.columns if c != "other_uid"]

        def sample_negs(uid: int, k: int) -> np.ndarray:
            positives = uid_to_pos.get(uid, set())
            banned = set(positives)
            banned.add(uid)
            # Candidates: allowed users excluding positives and self.
            # For typical 5k subsets, this is fast enough.
            candidates = np.array([u for u in allowed_py if u not in banned], dtype=np.int64)
            if candidates.size == 0:
                # Fallback: allow sampling from all allowed (excluding self)
                candidates = np.array([u for u in allowed_py if u != uid], dtype=np.int64)
            replace = candidates.size < k
            return rng.choice(candidates, size=k, replace=replace)

        for neg_name in neg_names:
            k = _infer_neg_k_from_filename(neg_name, default_k=default_k)
            rows: List[dict] = []
            for _, row in pos_df.iterrows():
                uid = int(row["uid"])
                negs = sample_negs(uid, k)
                base = {c: row[c] for c in base_cols}
                if "event_set" in base and isinstance(base["event_set"], str) and not base["event_set"].startswith("negative-"):
                    base["event_set"] = f"negative-{base['event_set']}"
                for other_uid in negs:
                    r = dict(base)
                    r["other_uid"] = int(other_uid)
                    rows.append(r)

            out_df = pd.DataFrame(rows, columns=base_cols + ["other_uid"])
            out_df.to_csv(os.path.join(output_dir, neg_name), index=False)


def discover_files(input_dir: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for name in os.listdir(input_dir):
        if not name.endswith(".csv"):
            continue
        files[name] = os.path.join(input_dir, name)
    return files


def main():
    p = argparse.ArgumentParser(description="Create a Brightkite friend_recommendation subset by users.")
    p.add_argument("--input_dir", default="..//data/tasks/brightkite/friend_recommendation", help="Directory containing CSVs (e.g., relational_train.csv, ...).")
    p.add_argument("--output_dir", default="..//data/tasks/brightkite/friend_recommendation_5000", help="Where to write subset CSVs.")
    p.add_argument("--num_users", type=int, default=5000, help="Number of users to keep.")
    p.add_argument("--method", choices=["snowball", "random"], default="snowball", help="User sampling method.")
    p.add_argument("--seed_users", type=int, default=50, help="Initial seed size for snowball sampling.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--chunksize", type=int, default=200_000, help="CSV read chunksize (0 to disable).")
    p.add_argument("--no_remap", action="store_true", help="Do not remap user ids to [0..num_users-1].")
    p.add_argument("--skip_regen_negatives", action="store_true",
                   help="Skip regenerating val/test negative-sample CSVs after subsetting.")
    p.add_argument("--default_neg_k", type=int, default=100,
                   help="Default K for filenames like relational_val_negative_sample.csv (no explicit _K suffix).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    files = discover_files(args.input_dir)

    # Prefer sampling users from *positive* relational edges only.
    positive_edge_names = ["relational_train.csv", "relational_val.csv", "relational_test.csv", "relational.csv"]
    positive_edge_files = [files[n] for n in positive_edge_names if n in files]
    if not positive_edge_files:
        raise SystemExit(
            f"No positive relational edge files found in {args.input_dir}. "
            f"Expected one of: {', '.join(positive_edge_names)}"
        )

    chunksize = None if args.chunksize <= 0 else args.chunksize

    if args.method == "random":
        all_users: Set[int] = set()
        for path in positive_edge_files:
            for chunk in _read_required_columns(path, usecols=["uid", "other_uid"], chunksize=chunksize):
                all_users.update(chunk["uid"].astype(int).unique().tolist())
                all_users.update(chunk["other_uid"].astype(int).unique().tolist())
        if len(all_users) < args.num_users:
            raise SystemExit(f"Only found {len(all_users)} users, cannot sample {args.num_users}.")
        selected_users = rng.sample(list(all_users), k=args.num_users)
    else:
        adj = build_undirected_adj(positive_edge_files, chunksize=chunksize)
        selected_users = snowball_users(adj, target_size=args.num_users, seed_size=args.seed_users, rng=rng)

    allowed_users = set(int(u) for u in selected_users)
    uid_map = None
    if not args.no_remap:
        uid_map = {old: new for new, old in enumerate(sorted(allowed_users))}

    os.makedirs(args.output_dir, exist_ok=True)

    # Write all CSVs in the directory, applying either relational or personal filtering rules.
    for filename, input_path in sorted(files.items()):
        output_path = os.path.join(args.output_dir, filename)
        is_relational = "relational" in filename and "personal" not in filename
        filter_and_write_csv(
            input_path=input_path,
            output_path=output_path,
            allowed_users=allowed_users,
            uid_map=uid_map,
            chunksize=chunksize,
            is_relational=is_relational,
        )

    meta = {
        "num_users": len(allowed_users),
        "method": args.method,
        "seed": args.seed,
        "seed_users": args.seed_users,
        "no_remap": bool(args.no_remap),
    }
    with open(os.path.join(args.output_dir, "subset_meta.json"), "w") as f:
        f.write(json.dumps(meta, indent=2))
        f.write("\n")

    if not args.skip_regen_negatives:
        regenerate_negative_samples(
            output_dir=args.output_dir,
            allowed_users=set(uid_map.values()) if uid_map is not None else allowed_users,
            seed=args.seed,
            default_k=args.default_neg_k,
        )


if __name__ == "__main__":
    main()