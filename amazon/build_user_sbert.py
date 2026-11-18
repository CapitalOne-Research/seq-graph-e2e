#!/usr/bin/env python3
"""Precompute the per-user pooled SBert vector that RelGT will use as a review-text feature.

Mirrors static_graph_clf.py exactly: reuse notebooks/review_embedding.py::process_and_embed_user_events
(warm all_embeddings.npy cache -> no SBert model / no blocked download) and masked-mean-pool over events.
Writes user_sbert.npy (row = DENSE uid, i.e. sorted-uid order == FraudDataset.uid2idx) into each
relational dir, where relbench_fraud_dataset.FraudDataset loads it under --use_review_text.

    python amazon/build_user_sbert.py --dataset clothing \
        --processed data/processed/amazon-clothing \
        --relational_glob 'data/tasks/amazon-clothing/fraud_detection_seed*/relational'
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "notebooks"))
from review_embedding import process_and_embed_user_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--processed", required=True, help="dir with all_nodes.csv (+ warm all_embeddings.npy cache)")
    ap.add_argument("--relational_glob", required=True,
                    help="glob of relational dirs to write user_sbert.npy into (all seeds share one array)")
    ap.add_argument("--max_events", type=int, default=50, help="must match static_graph_clf.py (default 50)")
    ap.add_argument("--sbert_path",
                    default=os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                    help="only touched on a cold cache; with a warm all_embeddings.npy it is never loaded")
    args = ap.parse_args()

    nodes = pd.read_csv(os.path.join(args.processed, "all_nodes.csv"))
    from sentence_transformers import SentenceTransformer  # lazy; only used on a cold cache
    emb, masks, _, uid_to_idx = process_and_embed_user_events(
        nodes, lambda: SentenceTransformer(args.sbert_path), args.max_events, args.processed)

    # masked mean pool over events -> one vector per user (identical to static_graph_clf.py)
    masks = np.asarray(masks).astype(bool)
    counts = np.clip(masks.sum(axis=1, keepdims=True), 1, None)
    pooled = (np.asarray(emb) * masks[..., None]).sum(axis=1) / counts   # (N_orig, D), row = uid_to_idx[uid]
    pooled = pooled.astype(np.float32)
    D = pooled.shape[1]

    rel_dirs = sorted(glob.glob(args.relational_glob))
    if not rel_dirs:
        raise SystemExit(f"no relational dirs matched: {args.relational_glob}")
    for rel in rel_dirs:
        users = pd.read_csv(os.path.join(rel, "users.csv"))
        dense_order = sorted(users["uid"].tolist())          # == FraudDataset.uid2idx ordering
        out = np.zeros((len(dense_order), D), dtype=np.float32)
        miss = 0
        for i, uid in enumerate(dense_order):
            j = uid_to_idx.get(uid)
            if j is None:
                miss += 1
            else:
                out[i] = pooled[j]
        np.save(os.path.join(rel, "user_sbert.npy"), out)
        print(f"wrote {rel}/user_sbert.npy  shape={out.shape}  (no-embedding fallback for {miss}/{len(dense_order)} users)")


if __name__ == "__main__":
    main()
