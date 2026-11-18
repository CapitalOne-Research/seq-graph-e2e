#!/usr/bin/env python3
"""Probe the COMPRESSED node vector z_i^{(0)} = Compress(Enc(X_i)) -- the pre-message-passing
pooled sequence embedding -- to test the mechanism claim in the paper's "Why does end-to-end
training help?" section.

The claim: end-to-end training routes the task gradient back through the pooling operation, so the
directions retained in the single compressed vector become task-/neighborhood-relevant, whereas a
task-agnostic (masked-LM, frozen) encoder shapes that vector to reconstruct events. This script
turns that hypothesis into a measurement, WITHOUT re-running any expensive training, by comparing
the SAME compressed vector produced by different encoders on two axes:

  (1) LINEAR LINK PROBE (headline). Freeze z, fit ONLY a linear scorer on the Hadamard product
      z_i (.) z_j over train edges (fresh random negatives), and rank held-out test friendships
      (1 positive vs K negatives, the paper's protocol). This is GRAPH-FREE: no GNN, no neighbor
      sampling. Because every arm gets an identical-capacity probe with identical init and negative
      samples (same seed), a higher probe MRR means the compressed vector ITSELF is more
      label-predictive -- which is exactly the mechanism's prediction, with the GNN and the
      trainable-capacity confound removed.
  (2) LINEAR CKA. How much did training reshape the representation geometry (CKA between two arms'
      z), and how well does each arm's z align with the graph's spectral (friendship) structure
      (CKA between z and the top eigenvectors of the normalized adjacency).

Cleanest comparison (isolates the task gradient, holding the MLM init fixed):
    frozen  = the frozen two-stage control  (run_controls.sh: run_tag=frozenseq_matched) -- MLM-pretrained, FROZEN
    joint   = the unfrozen two-stage control (run_controls.sh: run_tag=unfrozen2stage)  -- SAME MLM init, fine-tuned jointly
These two differ ONLY in whether the task gradient reaches the encoder, so joint>frozen on the
probe is direct evidence for the mechanism. Add the from-scratch end-to-end BRIDGE (no control tag)
as a third arm, and --include_raw for a random-init floor.

Usage:
  python probe_compressed.py \
      --ckpts frozen=brightkite_checkpoints_seeded/<...>_frozenseq_matched_best.pth \
              joint=brightkite_checkpoints_seeded/<...>_unfrozen2stage_best.pth \
              bridge=brightkite_checkpoints_seeded/<...>_best.pth \
      --include_raw --out probe_brightkite.json

Accepts both checkpoint formats: JointModel checkpoints (keys under `sequence_encoder.bert.*`) and
raw masked-LM checkpoints from sequence_embedding.py (keys under `bert.*`). Reuses joint.py's data
pipeline and metric, so node indexing / splits / MRR match the main results exactly. No training of
the encoders; only the tiny linear probe head is fit. CPU-friendly but fast on GPU.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig
from tqdm import tqdm

# Reuse the exact model/data/metric code from the training script so the probe is faithful.
from joint import (
    SequenceEncoder_mean,
    load_and_prepare_data,
    build_model_matching_ckpt,
    load_strict_or_filtered,
    evaluate,
    set_seed,
    generate_vocab,
    GeohashTokenizer,
)


# ------------------------------------------------------------------ data / inputs

def _namespace_from_ckpt(ckpt, overrides):
    """Reconstruct an args Namespace from a checkpoint's saved args, applying CLI overrides."""
    saved = dict(ckpt.get("args", {}) or {})
    for k, v in overrides.items():
        if v is not None:
            saved[k] = v
    saved.setdefault("debug_subset", None)
    saved.setdefault("event_set", "personal")
    saved.setdefault("num_negative_samples", 100)
    return argparse.Namespace(**saved)


def prepare_inputs(args, device):
    """Replicates joint.py's eval-only data prep: builds the graph + per-node tokenized sequences
    (indices 0..N-1). Returns (graph_data, seq_ids[N,L], seq_masks[N,L], vocab)."""
    graph_data, _uids, user_sequences_times = load_and_prepare_data(args)
    user_sequences = {uid: [e for e, t in seq] for uid, seq in user_sequences_times.items()}
    user_event_times = {uid: [t for e, t in seq] for uid, seq in user_sequences_times.items()}

    vocab = generate_vocab(args.csv_file, args.event_set)
    tokenizer = GeohashTokenizer(vocab)
    tokenizer.padding_side = "left"
    max_seq_tokens = args.max_events * 4 + 1

    all_ids = []
    for uid in tqdm(range(graph_data.num_nodes), desc="Tokenizing sequences"):
        seq = user_sequences.get(uid, [])
        input_ids = tokenizer.encode(seq, max_length=max_seq_tokens, padding="max_length", truncation=True)
        all_ids.append(input_ids)
    ids = torch.tensor(all_ids, dtype=torch.long).to(device)
    masks = (ids != vocab["[PAD]"]).long().to(device)
    return graph_data, ids, masks, vocab


# ------------------------------------------------------------------ encoders

def _bert_config(args, vocab_size):
    max_seq_tokens = args.max_events * 4 + 1
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.bert_hidden_size,
        num_hidden_layers=args.bert_num_layers,
        num_attention_heads=args.bert_num_heads,
        intermediate_size=args.bert_hidden_size * 2,
        max_position_embeddings=max_seq_tokens,
    )


def get_encoder(path, template_args, vocab_size, device):
    """Return the sequence encoder (eval mode) from a checkpoint. Handles JointModel checkpoints
    and raw masked-LM (sequence_embedding.py) checkpoints."""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("sequence_encoder.bert.") for k in sd):
        # JointModel checkpoint: rebuild the matching model from ITS OWN saved args, load weights.
        a = _namespace_from_ckpt(ckpt, {})
        model, _ = build_model_matching_ckpt(a, path)
        load_strict_or_filtered(model, ckpt)
        enc = model.sequence_encoder
    elif any(k.startswith("bert.") for k in sd):
        # Raw MLM checkpoint: transfer bert.* into a mean-pooling encoder matching the template.
        enc = SequenceEncoder_mean(_bert_config(template_args, vocab_size))
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        msg = enc.bert.load_state_dict(bert_sd, strict=False)
        print(f"  [mlm] loaded {len(bert_sd)} bert tensors (missing={len(msg.missing_keys)}, "
              f"unexpected={len(msg.unexpected_keys)})")
    else:
        raise ValueError(f"Unrecognized checkpoint format (no sequence_encoder.bert.* or bert.*): {path}")
    return enc.to(device).eval()


def random_encoder(template_args, vocab_size, device, seed=0):
    set_seed(seed)
    return SequenceEncoder_mean(_bert_config(template_args, vocab_size)).to(device).eval()


@torch.no_grad()
def encode_all_nodes(encoder, ids, masks, device, batch=512):
    """Return the pooled compressed vector z_i^{(0)} for every node -> [N, d]. If the encoder emits
    a token sequence [B,L,d] (stacked), reduce to a masked mean so we probe a single node vector.
    Sequences are LEFT-padded, so if this encoder's context is shorter than the shared inputs
    (checkpoints trained with different --max_events), keep the most recent tokens."""
    try:
        max_pos = int(encoder.bert.config.max_position_embeddings)
    except Exception:
        max_pos = ids.shape[1]
    if max_pos < ids.shape[1]:
        ids = ids[:, -max_pos:]
        masks = masks[:, -max_pos:]
    zs = []
    N = ids.shape[0]
    for i in tqdm(range(0, N, batch), desc="Encoding z^{(0)}", leave=False):
        b_ids = ids[i:i + batch]
        b_msk = masks[i:i + batch]
        out = encoder(b_ids, b_msk)
        if out.dim() == 3:  # [B, L, d] -> masked mean over valid tokens
            m = b_msk.unsqueeze(-1).float()
            out = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        zs.append(out.float().cpu())
    return torch.cat(zs, dim=0)


# ------------------------------------------------------------------ linear link probe

def _score_pairs(head, z, src, dst, chunk=2_000_000):
    outs = []
    for i in range(0, src.shape[0], chunk):
        s = src[i:i + chunk]
        d = dst[i:i + chunk]
        outs.append(head(z[s] * z[d]).squeeze(-1))
    return torch.cat(outs) if outs else torch.empty(0, device=z.device)


def link_probe(z, graph_data, hits_k, device, epochs, lr, seed, neg_ratio, train_cap):
    """Fit a LINEAR scorer on z_i (.) z_j over train edges; evaluate ranking on the test split.
    z is frozen. Identical seed => identical head init + negatives across arms, so only z differs."""
    set_seed(seed)  # identical probe init + negative draws across arms
    z = z.to(device)
    N, d = z.shape

    pos = graph_data.edge_index  # [2, E] (undirected)
    E = pos.shape[1]
    if train_cap and E > train_cap:
        sel = torch.randperm(E)[:train_cap]
        pos = pos[:, sel]
    pos = pos.to(device)
    P = pos.shape[1]

    head = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    bs = 100_000
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(P, device=device)
        for i in range(0, P, bs):
            idx = perm[i:i + bs]
            s, t = pos[0, idx], pos[1, idx]
            nneg = int(idx.numel() * neg_ratio)
            ns = torch.randint(0, N, (nneg,), device=device)
            nt = torch.randint(0, N, (nneg,), device=device)
            src = torch.cat([s, ns])
            dst = torch.cat([t, nt])
            logits = head(z[src] * z[dst]).squeeze(-1)
            labels = torch.cat([torch.ones(idx.numel(), device=device),
                                torch.zeros(nneg, device=device)])
            opt.zero_grad()
            bce(logits, labels).backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        pe = graph_data.test_edge_index.to(device)
        ne = graph_data.test_neg_edge_index.to(device)
        pos_scores = _score_pairs(head, z, pe[0], pe[1])
        neg_scores = _score_pairs(head, z, ne[0], ne[1])
        Pt = pos_scores.shape[0]
        K = neg_scores.shape[0] // Pt
        neg_scores = neg_scores[:Pt * K].view(Pt, K)
        candidates = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1).cpu()
        target = torch.zeros(Pt, dtype=torch.long)  # positive is always index 0
        metrics = evaluate(candidates, target, k_list=hits_k)
        try:
            from sklearn.metrics import roc_auc_score
            y = np.concatenate([np.ones(Pt), np.zeros(Pt * K)])
            s = np.concatenate([pos_scores.cpu().numpy(), neg_scores.reshape(-1).cpu().numpy()])
            metrics["AUC"] = float(roc_auc_score(y, s))
        except Exception:
            pass
    return metrics


# ------------------------------------------------------------------ CKA

def linear_cka(X, Y):
    """Linear CKA between [N,d1] and [N,d2] (scale/rotation invariant). Higher => more similar."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.t() @ Y).pow(2).sum()
    denom = (X.t() @ X).pow(2).sum().sqrt() * (Y.t() @ Y).pow(2).sum().sqrt()
    return float((hsic / denom.clamp(min=1e-12)).item())


def spectral_embedding(graph_data, k):
    """Top-k eigenvectors of the symmetric normalized adjacency (the graph's community structure).
    Returns [N, k] or None if scipy is unavailable / it fails."""
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        N = graph_data.num_nodes
        ei = graph_data.edge_index.cpu().numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)).tocsr()
        A = A.maximum(A.T)  # symmetrize
        deg = np.asarray(A.sum(1)).ravel()
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        D = sp.diags(dinv)
        An = D @ A @ D
        k = int(min(k, N - 2))
        vals, vecs = eigsh(An, k=k, which="LA")
        return torch.tensor(vecs, dtype=torch.float32)
    except Exception as e:
        print(f"  [spectral-cka] skipped ({type(e).__name__}: {e})")
        return None


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Linear-probe / CKA of the compressed node vector (frozen vs joint).")
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="label=path.pth entries. Recommended: frozen=<frozenseq_matched_best.pth> "
                         "joint=<unfrozen2stage_best.pth> bridge=<baseline_best.pth>")
    ap.add_argument("--include_raw", action="store_true", help="Add a random-init encoder as a floor.")
    ap.add_argument("--input_path", type=str, default=None, help="Override task dir (else from ckpt args).")
    ap.add_argument("--csv_file", type=str, default=None, help="Override personal.csv (else from ckpt args).")
    ap.add_argument("--hits_k", type=str, default="1,5,10,20")
    ap.add_argument("--probe_epochs", type=int, default=50)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_neg_ratio", type=float, default=2.0)
    ap.add_argument("--probe_seeds", type=str, default="0,1,2", help="Repeat the probe per seed for mean+/-std.")
    ap.add_argument("--train_cap", type=int, default=200_000, help="Cap train positive edges for probe speed.")
    ap.add_argument("--l2norm", action="store_true", default=True, help="L2-normalize z rows before probing (default on).")
    ap.add_argument("--no_l2norm", dest="l2norm", action="store_false")
    ap.add_argument("--spectral_k", type=int, default=128, help="Eigenvectors for spectral-alignment CKA (0=off).")
    ap.add_argument("--embed_batch", type=int, default=512)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out", type=str, default="probe_compressed.json")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    hits_k = [int(x) for x in args.hits_k.split(",")]
    seeds = [int(s) for s in args.probe_seeds.split(",")]

    arms = []
    for entry in args.ckpts:
        if "=" not in entry:
            raise SystemExit(f"--ckpts entries must be label=path (got '{entry}')")
        label, path = entry.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
        arms.append((label, path))

    # Data + tokenized inputs are built ONCE from the first checkpoint's args (all arms share the
    # dataset), so node ids / splits / negatives are identical across arms.
    template_ckpt = torch.load(arms[0][1], map_location="cpu")
    template_args = _namespace_from_ckpt(template_ckpt, {"input_path": args.input_path, "csv_file": args.csv_file})
    # Build shared inputs long enough for the longest-context encoder; each encoder is sliced down
    # to its own context in encode_all_nodes (checkpoints may differ in --max_events).
    me = int(getattr(template_args, "max_events", 50) or 50)
    for _lbl, _p in arms:
        if _p:
            _sa = (torch.load(_p, map_location="cpu").get("args", {}) or {})
            me = max(me, int(_sa.get("max_events", me) or me))
    template_args.max_events = me
    print(f"[data] preparing inputs from {arms[0][0]} args: dataset={getattr(template_args,'dataset','?')} "
          f"input_path={template_args.input_path} max_events={me}")
    graph_data, ids, masks, vocab = prepare_inputs(template_args, device)
    print(f"[data] N={graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} train edges, "
          f"{graph_data.test_edge_index.shape[1]} test positives, vocab={len(vocab)}")

    # Extract z^{(0)} per arm (+ optional raw floor).
    embeddings = {}
    max_pos_by_arm = {}
    if args.include_raw:
        arms = [("raw", None)] + arms
    for label, path in arms:
        print(f"[encode] {label} <- {os.path.basename(path) if path else 'random-init'}")
        enc = random_encoder(template_args, len(vocab), device) if path is None \
            else get_encoder(path, template_args, len(vocab), device)
        try:
            max_pos_by_arm[label] = int(enc.bert.config.max_position_embeddings)
        except Exception:
            max_pos_by_arm[label] = ids.shape[1]
        z = encode_all_nodes(enc, ids, masks, device, batch=args.embed_batch)
        if args.l2norm:
            z = torch.nn.functional.normalize(z, dim=1)
        embeddings[label] = z
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(set(max_pos_by_arm.values())) > 1:
        print(f"[warn] arms differ in context length (max_position_embeddings): {max_pos_by_arm}. "
              f"They were trained with different --max_events; shorter-context encoders see only the "
              f"most recent tokens. For the cleanest mechanism comparison use identically-configured arms.")

    # (1) Linear link probe, repeated over seeds.
    probe = {}
    for label, z in embeddings.items():
        runs = [link_probe(z, graph_data, hits_k, device, args.probe_epochs, args.probe_lr,
                           s, args.probe_neg_ratio, args.train_cap) for s in seeds]
        agg = {}
        for m in runs[0]:
            vals = np.array([r[m] for r in runs], dtype=float)
            agg[m] = {"mean": float(vals.mean()), "std": float(vals.std())}
        probe[label] = {"per_seed": runs, "agg": agg}

    # (2) CKA: pairwise between arms + alignment with graph spectral structure.
    labels = list(embeddings.keys())
    cka_pairwise = {}
    for a, b in itertools.combinations(labels, 2):
        cka_pairwise[f"{a}|{b}"] = linear_cka(embeddings[a], embeddings[b])
    cka_spectral = {}
    if args.spectral_k > 0:
        S = spectral_embedding(graph_data, args.spectral_k)
        if S is not None:
            for label, z in embeddings.items():
                cka_spectral[label] = linear_cka(z, S)

    # -------- report --------
    print("\n================  LINEAR LINK PROBE on the compressed vector z^{(0)}  ================")
    print("(graph-free: a linear scorer on z_i (.) z_j; identical probe across arms; mean+/-std over "
          f"seeds {seeds})")
    cols = ["MRR"] + [f"Hits@{k}" for k in hits_k] + (["AUC"] if "AUC" in probe[labels[0]]["agg"] else [])
    header = f"  {'arm':>8} | " + " | ".join(f"{c:^16}" for c in cols)
    print(header)
    for label in labels:
        a = probe[label]["agg"]
        cells = [f"{a[c]['mean']*100:6.2f}+/-{a[c]['std']*100:4.2f}" if c != "AUC"
                 else f"{a[c]['mean']:6.4f}+/-{a[c]['std']:5.4f}" for c in cols]
        print(f"  {label:>8} | " + " | ".join(f"{c:^16}" for c in cells))
    if "frozen" in probe and "joint" in probe:
        dm = (probe["joint"]["agg"]["MRR"]["mean"] - probe["frozen"]["agg"]["MRR"]["mean"]) * 100
        print(f"\n  joint - frozen  MRR delta = {dm:+.2f} pts  "
              f"(positive => the task gradient made the compressed vector more link-predictive; "
              f"this is the §5.4 mechanism, measured, with no GNN and no capacity change)")

    print("\n================  LINEAR CKA  ================")
    print("  pairwise (how much training reshaped z; 1=identical):")
    for k, v in cka_pairwise.items():
        print(f"    {k:>22}: {v:.4f}")
    if cka_spectral:
        print("  alignment with graph spectral structure (higher => z better reflects who-is-friends):")
        for k, v in cka_spectral.items():
            print(f"    {k:>22}: {v:.4f}")

    out = {
        "config": {k: getattr(args, k) for k in ["hits_k", "probe_epochs", "probe_lr",
                                                 "probe_neg_ratio", "probe_seeds", "train_cap",
                                                 "l2norm", "spectral_k"]},
        "dataset": getattr(template_args, "dataset", None),
        "num_nodes": int(graph_data.num_nodes),
        "arms": {label: (path or "random-init") for label, path in arms},
        "probe": probe,
        "cka_pairwise": cka_pairwise,
        "cka_spectral": cka_spectral,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
