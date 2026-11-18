#!/usr/bin/env python3
"""Convert a paper-cited 2014 Amazon review category into PRES-style events.

This is the command-line form of the retained Amazon processing notebook. It
also emits the review-helpfulness labels required by the fraud experiments.
"""
from __future__ import annotations

import argparse
import ast
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILES = {
    "clothing": "reviews_Clothing_Shoes_and_Jewelry.json.gz",
    "electronics": "reviews_Electronics.json.gz",
    "movies": "reviews_Movies_and_TV.json.gz",
    "musical": "reviews_Musical_Instruments.json.gz",
}


def read_reviews(path):
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid review record") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: review record is not a mapping")
            records.append(record)
    return pd.DataFrame.from_records(records)


def process(category, raw_dir, output_dir, min_reviews=5, min_shared_products=None):
    source = Path(raw_dir) / RAW_FILES[category]
    if not source.is_file():
        raise FileNotFoundError(source)
    required = {"reviewerID", "asin", "unixReviewTime", "overall", "reviewText", "helpful"}
    reviews = read_reviews(source)
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")

    counts = reviews.groupby("reviewerID")["asin"].count()
    reviews = reviews[reviews["reviewerID"].isin(counts[counts >= min_reviews].index)].copy()
    if reviews.empty:
        raise ValueError(f"{source}: no users have at least {min_reviews} reviews")
    reviews = reviews.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)
    reviews["rating"] = reviews["overall"].astype(int)
    reviews["uid"] = pd.factorize(reviews["reviewerID"])[0]

    personal = reviews.assign(
        timestamp=reviews["unixReviewTime"], event_set="personal",
        event=reviews["asin"].astype(str) + ":" + reviews["rating"].astype(str),
        other_uid=pd.NA,
    )[["uid", "timestamp", "event_set", "event", "reviewText", "other_uid"]]

    shared = 5 if category == "movies" else 3
    if min_shared_products is not None:
        shared = min_shared_products
    pairs = reviews[["reviewerID", "asin", "unixReviewTime"]].merge(
        reviews[["reviewerID", "asin", "unixReviewTime"]], on="asin", suffixes=("_x", "_y")
    )
    pairs["timestamp"] = np.maximum(pairs["unixReviewTime_x"], pairs["unixReviewTime_y"])
    pairs = pairs.groupby(["reviewerID_x", "reviewerID_y"], as_index=False).agg(
        shared_products=("asin", "count"), timestamp=("timestamp", "min")
    )
    pairs = pairs[(pairs["reviewerID_x"] != pairs["reviewerID_y"])
                  & (pairs["shared_products"] >= shared)]
    user_ids = reviews[["reviewerID", "uid"]].drop_duplicates()
    pairs = pairs.merge(user_ids, left_on="reviewerID_x", right_on="reviewerID")
    pairs = pairs.merge(
        user_ids.rename(columns={"reviewerID": "reviewerID_y", "uid": "other_uid"}),
        on="reviewerID_y",
    )
    relational = pairs.assign(event_set="relational", event="co-review_product")[
        ["uid", "timestamp", "event_set", "event", "other_uid"]
    ]

    events = pd.concat([personal, relational], ignore_index=True).drop_duplicates()
    events = events.sort_values(["uid", "timestamp", "event_set", "other_uid"], na_position="last")
    event_sets = events.groupby("uid")["event_set"].nunique()
    accounts_with_both = event_sets[event_sets == 2].index
    combined = events[events["uid"].isin(accounts_with_both)]

    helpful = pd.DataFrame(reviews["helpful"].tolist(), columns=["helpful_num", "helpful_den"])
    helpful["uid"] = reviews["uid"].to_numpy()
    labels = helpful.groupby("uid", as_index=False)[["helpful_num", "helpful_den"]].sum()
    labels["helpful_prop"] = labels["helpful_num"] / labels["helpful_den"]
    labels["fraud_label"] = -1
    labels.loc[(labels["helpful_prop"] < .3) & (labels["helpful_den"] >= min_reviews), "fraud_label"] = 1
    labels.loc[(labels["helpful_prop"] > .7) & (labels["helpful_den"] >= min_reviews), "fraud_label"] = 0
    labels = labels[labels["fraud_label"] != -1][["uid", "fraud_label"]]
    combined_labels = labels[labels["uid"].isin(accounts_with_both)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"amazon-{category}"
    outputs = {
        "all_events": Path(str(prefix) + "_all_events.csv"),
        "accounts_with_pr": Path(str(prefix) + "_accounts_with_pr.csv"),
        "all_events_label": Path(str(prefix) + "_all_events_label.csv"),
        "accounts_with_pr_label": Path(str(prefix) + "_accounts_with_pr_label.csv"),
    }
    events.to_csv(outputs["all_events"], index=False)
    combined.to_csv(outputs["accounts_with_pr"], index=False)
    labels.to_csv(outputs["all_events_label"], index=False)
    combined_labels.to_csv(outputs["accounts_with_pr_label"], index=False)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=RAW_FILES)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-reviews", type=int, default=5)
    parser.add_argument("--min-shared-products", type=int)
    args = parser.parse_args(argv)
    outputs = process(args.category, args.raw_dir, args.output_dir,
                      args.min_reviews, args.min_shared_products)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
