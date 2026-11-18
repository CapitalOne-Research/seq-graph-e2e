#!/usr/bin/env python3
"""Trivial-baseline audit for the Amazon fraud-detection (node classification) task.

Fraud is a highly imbalanced per-user binary label (~2.5% positive). The question is NOT
label-in-inputs leakage (as with co-review) but TASK TRIVIALITY: can a simple hand-feature
classifier match the trained sequence+graph model? Shill/fake-review fraud is often trivially
separable by activity volume, rating extremity, and burstiness. If a logistic regression on a
handful of hand features rivals the trained model's AUC, the fraud task does not corroborate the
"sequence+graph modeling helps" story. If trivial AUC is well below trained AUC, fraud is a
genuine, co-location-free corroboration that de-risks the paper.

Baselines reported: majority-class (AUC=0.5 by construction) and a class-balanced logistic
regression on per-user hand features. Also reports each feature's univariate AUC (which single
trivial signal, if any, carries the task) as a leakage/triviality probe.

No deep learning. pandas + scikit-learn.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

DATASETS = ["clothing", "musical", "home-kitchen", "electronics"]


def rating_of(event):
    s = str(event)
    if ":" in s:
        try:
            return float(s.rsplit(":", 1)[1])
        except ValueError:
            return np.nan
    return np.nan


def build_features(events_csv):
    """Per-uid hand features from the event log (no sequence/graph modeling)."""
    df = pd.read_csv(events_csv, usecols=["uid", "timestamp", "event_set", "event"])
    per = df[df["event_set"] == "personal"].copy()
    rel = df[df["event_set"] == "relational"]

    per["rating"] = per["event"].map(rating_of)
    per["asin"] = per["event"].astype(str).str.split(":", n=1).str[0]

    g = per.groupby("uid")
    feats = pd.DataFrame({
        "n_events": g.size(),
        "n_distinct_asin": g["asin"].nunique(),
        "mean_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "frac_5": g["rating"].apply(lambda r: float((r == 5).mean())),
        "frac_1": g["rating"].apply(lambda r: float((r == 1).mean())),
        "ts_span_days": g["timestamp"].apply(lambda t: (t.max() - t.min()) / 86400.0),
        "n_unique_ts": g["timestamp"].nunique(),
    })
    feats["repeat_ratio"] = 1.0 - (feats["n_distinct_asin"] / feats["n_events"].clip(lower=1))
    feats["events_per_ts"] = feats["n_events"] / feats["n_unique_ts"].clip(lower=1)  # burstiness
    feats["events_per_day"] = feats["n_events"] / feats["ts_span_days"].clip(lower=1.0)
    feats["frac_extreme"] = feats["frac_1"] + feats["frac_5"]
    rel_deg = rel.groupby("uid").size().rename("n_relational")
    feats = feats.join(rel_deg).fillna({"n_relational": 0.0})
    return feats.fillna(0.0)


def auc_ap(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return roc_auc_score(y, s), average_precision_score(y, s)


def evaluate(task_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    feats = build_features(os.path.join(task_dir, "events.csv"))
    tr = pd.read_csv(os.path.join(task_dir, "label_train.csv")).set_index("uid")
    te = pd.read_csv(os.path.join(task_dir, "label_test.csv")).set_index("uid")

    cols = list(feats.columns)
    Xtr = feats.reindex(tr.index).fillna(0.0)[cols].to_numpy()
    Xte = feats.reindex(te.index).fillna(0.0)[cols].to_numpy()
    ytr = tr["fraud_label"].to_numpy()
    yte = te["fraud_label"].to_numpy()

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    auc, ap = auc_ap(yte, s)

    # univariate AUC per feature (which single trivial signal carries the task)
    uni = []
    for j, c in enumerate(cols):
        v = Xte[:, j]
        try:
            a = roc_auc_score(yte, v)
        except ValueError:
            a = 0.5
        uni.append((c, max(a, 1 - a)))  # direction-agnostic
    uni.sort(key=lambda x: -x[1])
    return dict(auc=auc, ap=ap, base_rate=float(yte.mean()),
               n_test=len(yte), n_pos=int(yte.sum()), top_uni=uni[:5], cols=cols)


def main():
    ap = argparse.ArgumentParser(description="Fraud trivial-baseline audit")
    ap.add_argument("--data_root", default="../data/tasks")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--task", default="fraud_detection")
    args = ap.parse_args()

    try:
        import sklearn  # noqa
    except ImportError:
        sys.exit("scikit-learn required (pip install scikit-learn).")

    any_ok = False
    for ds in args.datasets:
        task_dir = os.path.join(args.data_root, f"amazon-{ds}", args.task)
        if not os.path.isdir(task_dir):
            print(f"[skip] {ds}: {task_dir} not found")
            continue
        any_ok = True
        r = evaluate(task_dir)
        print(f"\n===== amazon-{ds} fraud ({r['n_test']} test users, {r['n_pos']} fraud, "
              f"base rate {r['base_rate']*100:.2f}%) =====")
        print(f"  majority-class:            ROC-AUC=0.500")
        print(f"  trivial hand-feature LR:   ROC-AUC={r['auc']:.4f}   AP={r['ap']:.4f}  (AP baseline={r['base_rate']:.4f})")
        print("  top univariate feature AUCs (direction-agnostic):")
        for c, a in r["top_uni"]:
            print(f"     {c:>16}: {a:.4f}")

    if not any_ok:
        sys.exit("No fraud task dirs found.")
    print("\nInterpretation: compare 'trivial hand-feature LR' ROC-AUC to the TRAINED sequence+graph")
    print("model's reported fraud AUC. If close, fraud is a trivial-feature task and does not")
    print("corroborate the sequence+graph story. If the trained model is clearly higher, fraud is a")
    print("clean, co-location-free corroboration. A single univariate AUC near the trained number")
    print("would indicate the task hinges on one trivial signal (e.g., rating extremity/burstiness).")


if __name__ == "__main__":
    main()
