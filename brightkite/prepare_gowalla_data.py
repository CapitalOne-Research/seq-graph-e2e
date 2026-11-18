#!/usr/bin/env python3
"""Download the Gowalla friend-recommendation task from a supplied base URL and build the
100-negatives-per-positive evaluation files that joint.py reads (relational_{val,test}_negative_sample_100.csv).

The HF negative files hold a block of 1000 negatives per positive edge, in the same order as the positives
(rows of relational_{split}.csv). We keep the first 100 rows of each block. The relational files have an
empty timestamp and constant event_set/event, so blocks cannot be told apart by their columns; they are
aligned by order, and every block's uid is checked against its positive's uid.

The full negative files (8.4 GB val, 16.8 GB test) are streamed, never stored.

    python prepare_gowalla_data.py --base-url URL        # writes ../data/tasks/gowalla/friend_recommendation/
    python prepare_gowalla_data.py --max_positives 1000  # quick format check on a prefix
"""
import argparse
import os
import subprocess
import sys

PLAIN = ["personal.csv", "relational_train.csv", "relational_val.csv", "relational_test.csv"]


def download(name, out_dir, base_url):
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        print(f"[skip] {name} exists")
        return path
    print(f"[download] {name}")
    subprocess.run(["curl", "-fsSL", "-o", path + ".part", f"{base_url}/{name}"], check=True)
    os.replace(path + ".part", path)
    return path


def positive_uids(path, max_positives):
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        assert header[0] == "uid" and header[-1] == "other_uid", header
        uids = [line.split(",", 1)[0] for line in f]
    return uids[:max_positives] if max_positives else uids


def build_split(split, out_dir, k, block, max_positives, base_url):
    out = os.path.join(out_dir, f"relational_{split}_negative_sample_{k}.csv")
    if os.path.exists(out) and not max_positives:
        print(f"[skip] {os.path.basename(out)} exists")
        return
    uids = positive_uids(os.path.join(out_dir, f"relational_{split}.csv"), max_positives)
    n = len(uids)
    print(f"[stream] relational_{split}_negative_sample.csv -> {os.path.basename(out)} ({n} positives x {k})")
    proc = subprocess.Popen(["curl", "-fsSL", f"{base_url}/relational_{split}_negative_sample.csv"], stdout=subprocess.PIPE)
    r = proc.stdout
    with open(out + ".part", "w") as w:
        header = r.readline().decode()
        assert header.rstrip("\n").split(",")[-1] == "other_uid", header
        w.write(header)
        i = 0  # row index within the negative file
        for raw in r:
            pos, j = divmod(i, block)
            if pos >= n:
                break
            line = raw.decode()
            uid = line.split(",", 1)[0]
            if uid != uids[pos]:
                sys.exit(f"[error] {split}: negative row {i} has uid {uid}, positive {pos} has uid {uids[pos]}; "
                         f"blocks are not {block} rows per positive")
            if j < k:
                w.write(line)
            i += 1
            if i % 20_000_000 == 0:
                print(f"  {i:,} rows read ({pos:,}/{n:,} positives)", flush=True)
        if i != n * block:
            sys.exit(f"[error] {split}: read {i:,} negative rows, expected {n * block:,} ({n} x {block})")
    proc.kill()  # stops the download early when --max_positives is set
    proc.wait()
    os.replace(out + ".part", out)
    print(f"[ok] {os.path.basename(out)}: {n * k:,} rows ({k} per positive, aligned)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("GOWALLA_DATA_URL"),
                    help="base URL for the task CSVs; alternatively set GOWALLA_DATA_URL")
    ap.add_argument("--out_dir", default="../data/tasks/gowalla/friend_recommendation")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--block", type=int, default=1000, help="negatives per positive in the HF files")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--max_positives", type=int, default=0, help="only the first N positives (format check)")
    a = ap.parse_args()
    if not a.base_url or not a.base_url.startswith(("https://", "http://")):
        ap.error("Provide an HTTP(S) --base-url or set GOWALLA_DATA_URL")
    a.base_url = a.base_url.rstrip("/")
    if not 0 < a.k <= a.block or a.max_positives < 0:
        ap.error("Require 0 < k <= block and max_positives >= 0")
    splits = a.splits.split(",")
    if any(split not in ("val", "test") for split in splits):
        ap.error("--splits must contain val and/or test")
    os.makedirs(a.out_dir, exist_ok=True)
    for name in PLAIN:
        download(name, a.out_dir, a.base_url)
    for split in splits:
        build_split(split, a.out_dir, a.k, a.block, a.max_positives, a.base_url)


if __name__ == "__main__":
    main()
