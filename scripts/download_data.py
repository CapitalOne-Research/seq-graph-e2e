#!/usr/bin/env python3
"""Download the paper's PRES release and optional raw Amazon review inputs.

The PRES release is pinned to the revision used when this repository was
prepared. Raw Amazon files are the 2014 per-category review files cited by
arXiv:2510.11903; they are governed by the terms linked from the UCSD page.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import urllib.request

PRES_REPO_ENV = "PRES_REPO"
PRES_REVISION = "6f0a75d92c29586b7179674226ae4bc466a11159"
PRES_DATASETS = (
    "amazon-clothing",
    "amazon-electronics",
    "brightkite",
    "github",
    "gowalla",
)
AMAZON_URLS = {
    "movies": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV.json.gz",
    "electronics": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz",
    "clothing": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry.json.gz",
}


def expand(values, available):
    if not values:
        return []
    return list(available) if "all" in values else values


def pres_patterns(datasets):
    patterns = []
    for dataset in datasets:
        patterns.extend((f"processed/{dataset}_all_events.csv", f"tasks/{dataset}/**"))
    return patterns


def download_url(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"exists: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as source, partial.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"downloaded: {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pres-datasets", nargs="+", default=["all"],
        choices=["all", *PRES_DATASETS],
        help="PRES datasets to fetch (default: all five published datasets)",
    )
    parser.add_argument(
        "--raw-amazon", nargs="*", default=[],
        choices=["all", *AMAZON_URLS],
        help="Additional 2014 raw categories; 'all' means the three reported Amazon experiments",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--pres-repo", default=os.environ.get(PRES_REPO_ENV),
        help=(
            "Hugging Face dataset repo id for the PRES release (see the "
            f"companion paper, arXiv:2510.11903). Falls back to the "
            f"{PRES_REPO_ENV} environment variable."
        ),
    )
    parser.add_argument("--accept-amazon-license", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pres = expand(args.pres_datasets, PRES_DATASETS)
    raw = expand(args.raw_amazon, AMAZON_URLS)
    patterns = pres_patterns(pres)
    print(f"PRES revision: {PRES_REVISION}")
    for pattern in patterns:
        print(f"PRES: {pattern}")
    for category in raw:
        url = AMAZON_URLS[category]
        print(f"Amazon {category}: {url} -> {args.data_dir / 'raw' / Path(url).name}")
    if args.dry_run:
        return 0
    if raw and not args.accept_amazon_license:
        parser.error("raw Amazon downloads require --accept-amazon-license; review the paper-cited UCSD terms first")

    if patterns:
        if not args.pres_repo:
            parser.error(
                f"--pres-repo (or {PRES_REPO_ENV}) is required to download PRES; "
                "see the companion paper, arXiv:2510.11903, for the dataset repo id"
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            parser.error("install huggingface_hub to download PRES")
        snapshot_download(
            repo_id=args.pres_repo,
            repo_type="dataset",
            revision=PRES_REVISION,
            allow_patterns=patterns,
            local_dir=args.data_dir,
        )
    for category in raw:
        url = AMAZON_URLS[category]
        download_url(url, args.data_dir / "raw" / Path(url).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
