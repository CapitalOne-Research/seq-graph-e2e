#!/usr/bin/env python3
"""Read-only validation of canonical fraud CSVs and/or cached temporal .pt splits.

CSV-only checks use the Python standard library. Torch is needed only when a cached
.pt is present. This command never creates, replaces, or deletes split files.
"""
import argparse
import csv
from pathlib import Path

SPLITS = ("train", "val", "test")


def uid(value):
    """Match pandas' integer UID inference used by the Amazon preprocessing."""
    value = value.strip()
    if not value:
        raise ValueError("empty user ID")
    try:
        return int(value)
    except ValueError:
        return value


def rows(path, required):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {', '.join(required)}")
        return list(reader)


def csv_split(path):
    result = {}
    for row in rows(path, ("uid", "split")):
        user, split = uid(row["uid"]), row["split"]
        if user in result:
            raise ValueError(f"{path}: duplicate user {user} (overlapping split rows)")
        if split not in SPLITS:
            raise ValueError(f"{path}: invalid split {split!r} for user {user}")
        result[user] = split
    return result


def temporal_split(path, users):
    import torch  # optional for CSV-only validation

    cached = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    for split in SPLITS:
        mask = cached[f"{split}_mask"].bool().cpu().ravel().tolist()
        if len(mask) < len(users):
            raise ValueError(f"{path}: {split}_mask shorter than temporal user block")
        for user, selected in zip(users, mask):
            if selected:
                if user in result:
                    raise ValueError(f"{path}: overlapping masks for user {user}")
                result[user] = split
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip seeds with neither CSV nor .pt; at least one must validate")
    args = parser.parse_args()
    base = Path(args.processed)
    try:
        nodes = rows(base / "all_nodes.csv", ("uid",))
        edges = rows(base / "all_edges.csv", ("uid", "other_uid"))
        labeled = {uid(row["uid"]) for row in nodes}
        users = sorted(labeled | {uid(row[key]) for row in edges for key in ("uid", "other_uid")})
        if not labeled:
            raise ValueError("no labeled users in all_nodes.csv")
    except (OSError, ValueError, TypeError) as error:
        print(f"RESULT: FAIL - {error}")
        return 1

    valid, failed = 0, False
    for seed in args.seeds:
        prefix = f"[{args.dataset} seed{seed}]"
        canonical = base / "splits" / f"{args.dataset}_canonical_user_split_seed{seed}.csv"
        cached = base / "splits" / f"{args.dataset}_splits_seed{seed}.pt"
        if not canonical.is_file() and not cached.is_file():
            print(f"{prefix} {'SKIP' if args.allow_missing else 'FAIL'} - missing canonical CSV and temporal .pt")
            failed |= not args.allow_missing
            continue
        try:
            split = csv_split(canonical) if canonical.is_file() else None
            temporal = temporal_split(cached, users) if cached.is_file() else None
            if split is not None and temporal is not None and split != temporal:
                raise ValueError("canonical CSV differs from cached temporal user assignments")
            split = split if split is not None else temporal
            missing = labeled - split.keys()
            unknown = split.keys() - set(users)
            if missing or unknown:
                raise ValueError(f"coverage mismatch: {len(missing)} labeled users missing, {len(unknown)} unknown users")
            counts = [sum(split[user] == name for user in labeled) for name in SPLITS]
            print(f"{prefix} PASS - labeled_users={len(labeled)} train/val/test={'/'.join(map(str, counts))}"
                  + ("; CSV/temporal identity PASS" if canonical.is_file() and cached.is_file() else ""))
            valid += 1
        except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
            print(f"{prefix} FAIL - {error}")
            failed = True
    ok = valid > 0 and not failed
    print(f"RESULT: {'PASS' if ok else 'FAIL'} - {valid} seed(s) verified (read-only)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
