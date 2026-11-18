import os
import time
import argparse
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from sentence_transformers import SentenceTransformer
from types import SimpleNamespace

from review_embedding import LocalLogger, MLPClassifier, process_and_embed_user_events
from temporal_graph import load_data, evaluate, set_seed

# Make TPNet repo root importable (TPNet/ is at the repo root)
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from TPNet.TPNet import TPNet, RandomProjectionModule  # type: ignore
from TPNet.dygformer import DyGFormer  # type: ignore
from TPNet.graph_mixer import GraphMixer  # type: ignore
from TPNet.cawn import CAWN  # type: ignore
from TPNet.tgat import TGAT  # type: ignore
from TPNet.tcl import TCL  # type: ignore
from TPNet.nat import NAT  # type: ignore
from TPNet.ctan.ctdg_models import CTAN  # type: ignore
from TPNet.utils import NeighborSampler  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fraud node classification with temporal encoders from the TPNet pipeline"
    )
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"),
                        help="dir with amazon-<dataset>/ (all_nodes.csv, all_edges.csv, splits/)")
    parser.add_argument("--log_dir", type=str, default="logs_delta",
                        help="root dir for LocalLogger JSONs; use a fresh dir for the canonical rerun")
    parser.add_argument(
        "--model",
        type=str,
        choices=["TPNet", "DyGFormer", "GraphMixer", "CAWN", "TGAT", "TCL", "NAT", "CTAN"],
        default="TPNet",
        help="Temporal encoder to use for node classification.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["musical", "movies", "clothing", "electronics", "home-kitchen", "office-products"],
        default="musical",
    )
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=2 ** 12)
    parser.add_argument("--eval_batch_size", type=int, default=2 ** 18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_events",
        type=int,
        default=50,
        help="Maximum number of events per user (for SBERT sequence embedding).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (in epochs).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Evaluation frequency (in epochs).",
    )
    parser.add_argument(
        "--enable_early_stopping",
        action="store_true",
        help="Enable early stopping based on validation PR-AUC.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--clf_layers", type=int, default=2)
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=2048,
        help="Batch size when computing node embeddings over nodes.",
    )
    parser.add_argument(
        "--force_new_splits",
        action="store_true",
        default=False,
        help="Force creation of new node splits instead of reusing cached ones.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="TPNet-fraud",
        help="Experiment name for saving checkpoints and logs.",
    )
    parser.add_argument(
        "--use_sequence_embedding",
        action="store_true",
        help="Initialize TPNet-family node features for users from SBERT review sequences.",
    )

    # TPNet random-projection hyperparameters (mirroring amazon/temporal_graph_new.py defaults)
    parser.add_argument(
        "--rp_dim_factor",
        type=int,
        default=2,
        help="Factor for random projection dim (≈ rp_dim_factor * log(2 * |E|)).",
    )
    parser.add_argument(
        "--rp_num_layer",
        type=int,
        default=2,
        help="Number of temporal walk layers in random projections.",
    )
    parser.add_argument(
        "--rp_time_decay_weight",
        type=float,
        default=1e-4,
        help="Time decay weight λ for temporal walks.",
    )
    parser.add_argument(
        "--rp_use_matrix",
        action="store_true",
        help="If set, store temporal walk matrices explicitly instead of random projections.",
    )
    parser.add_argument(
        "--rp_not_scale",
        action="store_true",
        help="If set, do not scale inner products of random projections.",
    )
    parser.add_argument(
        "--rp_enforce_dim",
        type=int,
        default=-1,
        help="If not -1, explicitly set random projection dimension to this value.",
    )

    return parser.parse_args()


def build_tpnet_neighbor_sampler(train_data, num_nodes, seed, sample_strategy="uniform", time_scaling=0.0):
    """
    Construct TPNet's NeighborSampler from chronological training edges.
    This mirrors the helper in amazon/temporal_graph_new.py.
    """
    num_nodes_with_pad = num_nodes + 1  # reserve index 0 for padding
    adj_list = [[] for _ in range(num_nodes_with_pad)]

    src_np = train_data.src.cpu().numpy()
    dst_np = train_data.dst.cpu().numpy()
    t_np = train_data.t.cpu().numpy()

    for edge_idx, (src, dst, ts) in enumerate(zip(src_np, dst_np, t_np), start=1):
        src_id = int(src) + 1
        dst_id = int(dst) + 1
        adj_list[src_id].append((dst_id, edge_idx, ts))
        adj_list[dst_id].append((src_id, edge_idx, ts))

    neighbor_sampler = NeighborSampler(
        adj_list,
        sample_neighbor_strategy=sample_strategy,
        time_scaling_factor=time_scaling,
        seed=seed,
    )
    return neighbor_sampler, len(src_np)


def build_tpnet_features(num_nodes, edge_num, feat_dim, seed):
    """
    Create deterministic random node/edge features for TPNet.
    Mirrors amazon/temporal_graph_new.py.
    """
    rng = np.random.default_rng(seed)
    node_features = rng.standard_normal((num_nodes + 1, feat_dim)).astype(np.float32)
    edge_features = rng.standard_normal((edge_num + 1, feat_dim)).astype(np.float32)
    node_features[0] = 0.0
    edge_features[0] = 0.0
    return node_features, edge_features


def init_tpnet_user_features_from_reviews(data, node_features, data_dir, args):
    """
    Initialize TPNet-style node_raw_features for user nodes with SBERT-based
    review representations (as in review_embedding.py and temporal_graph.py).
    """
    num_users = getattr(data, "num_users", None)
    user_ids = getattr(data, "user_ids", None)
    if num_users is None or user_ids is None:
        logging.warning(
            "Data object does not have `num_users`/`user_ids`; "
            "skipping SBERT-based initialization for TPNet features."
        )
        return

    try:
        nodes_df = pd.read_csv(os.path.join(data_dir, "all_nodes.csv"))
        sbert_model_path = "sentence-transformers/all-MiniLM-L6-v2"
        sbert_model = SentenceTransformer(sbert_model_path)

        emb_array, mask_array, _, uid_to_idx = process_and_embed_user_events(
            nodes_df, sbert_model, args.max_events, data_dir
        )
    except Exception as e:
        logging.warning(
            "Could not load or build SBERT review embeddings for TPNet features (%s); "
            "falling back to random node features.",
            e,
        )
        return

    N_all, _, D = emb_array.shape
    logging.info(
        "Loaded SBERT review embeddings for TPNet features with shape (N=%d, T=?, D=%d). "
        "Building per-user pooled representations for %d users.",
        N_all,
        D,
        num_users,
    )

    user_text_emb = np.zeros((num_users, D), dtype=np.float32)
    for local_idx, uid in enumerate(user_ids):
        idx = uid_to_idx.get(uid, None)
        if idx is None:
            continue
        seq_emb = emb_array[idx]
        seq_mask = mask_array[idx]
        if seq_mask.any():
            user_text_emb[local_idx] = seq_emb[seq_mask].mean(axis=0)

    embedding_dim = node_features.shape[1]
    if D != embedding_dim:
        logging.info(
            "Projecting SBERT dimension D=%d to embedding_dim=%d for TPNet features.",
            D,
            embedding_dim,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proj = torch.nn.Linear(D, embedding_dim, bias=False).to(device)
        torch.nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            projected = proj(torch.from_numpy(user_text_emb).to(device)).cpu().numpy()
        user_text_emb = projected

    # TPNet node_features is (num_nodes + 1, dim) with index 0 as padding,
    # and user node ids are 0..num_users-1. Shift by +1 when writing.
    node_features[1 : num_users + 1, :] = user_text_emb
    logging.info(
        "Initialized TPNet node_raw_features[1:%d] with SBERT-based user review embeddings.",
        num_users + 1,
    )


@torch.no_grad()
def compute_node_embeddings_temporal(encoder, data, train_data, device, args):
    """
    Compute one embedding per node using the chosen temporal encoder.
    We query embeddings at a common reference time (the max timestamp),
    and only for user nodes (which are the ones with labels).
    """
    num_users = getattr(data, "num_users", data.num_nodes)
    t_ref = float(data.t.max().item())

    if args.model == "NAT":
        encoder.eval()
        encoder.init_ncache()
        encoder.reset_random_state()

        train_loader = TemporalDataLoader(train_data, batch_size=args.train_batch_size)
        for batch in train_loader:
            batch = batch.to(device)
            src_np = (batch.src + 1).cpu().numpy().astype(np.int64)
            dst_np = (batch.dst + 1).cpu().numpy().astype(np.int64)
            t_np = batch.t.cpu().numpy().astype(np.float64)
            e_id_np = batch.e_id.cpu().numpy().astype(np.int64)
            encoder.compute_edge_temporal_embeddings(
                src_np, dst_np, t_np, e_id_np, edges_are_positive=True
            )

        return encoder.self_rep[1 : num_users + 1].detach()

    if args.model == "CTAN":
        encoder.eval()
        encoder.reset_memory()

        assoc = torch.empty(data.num_nodes + 1, dtype=torch.long, device=device)
        train_loader = TemporalDataLoader(train_data, batch_size=args.train_batch_size)

        for batch in train_loader:
            batch = batch.to(device)
            src = batch.src + 1
            dst = batch.dst + 1
            t = batch.t
            msg = batch.msg

            n_id = torch.cat([src, dst], dim=0).unique()
            assoc[n_id] = torch.arange(n_id.numel(), device=device)
            edge_index = torch.stack([assoc[src], assoc[dst]], dim=0)

            batch_ns = SimpleNamespace(src=src, dst=dst, t=t)
            _, _, emb_src, emb_pos_dst = encoder(batch_ns, n_id, msg, t, edge_index, assoc)
            encoder.update(src, dst, t, msg, emb_src, emb_pos_dst)

        z_nodes, _ = encoder.memory(torch.arange(1, num_users + 1, device=device))
        return z_nodes.detach()

    # All supported encoders use 1-based node ids with 0 reserved for padding.
    batch_size = max(1, int(getattr(args, "embed_batch_size", 2048)))
    emb_chunks = []

    extra_kwargs = {}
    if args.model in ["GraphMixer"]:
        # Align number of tokens in the mixer with the number of neighbors sampled.
        extra_kwargs = {"num_neighbors": args.num_neighbors}
    elif args.model in ["CAWN"]:
        extra_kwargs = {}
    elif args.model in ["TGAT"]:
        extra_kwargs = {"num_neighbors": args.num_neighbors}
    elif args.model in ["TCL"]:
        extra_kwargs = {"num_neighbors": args.num_neighbors}

    for start in range(0, num_users, batch_size):
        end = min(num_users, start + batch_size)
        node_ids = np.arange(start, end, dtype=np.int64)
        times_np = np.full_like(node_ids, t_ref, dtype=np.float64)
        src_ids = node_ids + 1
        dst_ids = node_ids + 1

        src_emb, _ = encoder.compute_src_dst_node_temporal_embeddings(
            src_ids, dst_ids, times_np, **extra_kwargs
        )
        emb_chunks.append(src_emb)

    return torch.cat(emb_chunks, dim=0)  # (num_users, embedding_dim)


@torch.no_grad()
def evaluate_split(encoder, nodeclf, data, train_data, device, args, split="val"):
    """
    Evaluate node classification performance on the given split.
    Uses the same evaluate() helper as notebooks/temporal_graph.py.
    """
    encoder.eval()
    nodeclf.eval()

    z_nodes = compute_node_embeddings_temporal(encoder, data, train_data, device, args)  # torch.Tensor
    logits = nodeclf(z_nodes.to(device))

    num_users = z_nodes.size(0)
    mask = getattr(data, f"{split}_mask") & (data.y != -1)
    mask_users = mask[:num_users]
    labels = data.y[:num_users][mask_users]
    split_logits = logits[:num_users][mask_users]

    loss = F.cross_entropy(split_logits, labels)
    probs = F.softmax(split_logits, dim=1)[:, 1]

    metrics = evaluate(probs.cpu(), labels.cpu())
    metrics["loss"] = float(loss.item())
    return metrics


def train_temporal_fraud(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    run_name = (
        f"{args.model}_hidden{args.embedding_dim}_"
        f"lr{args.lr}_bs{args.train_batch_size}_epochs{args.epochs}_seed{args.seed}"
    )
    project_name = f"temporal-graph-fraud-{args.dataset}"
    if args.use_sequence_embedding and not run_name.endswith("_sbert"):
        run_name += "_sbert"
    logger = LocalLogger(project_name=project_name, run_name=run_name, log_dir=args.log_dir)
    if logger.skip:
        print("Exiting script because the experiment is being skipped.")
        return

    # Data directory consistent with notebooks/temporal_graph.py
    data_dir = os.path.join(args.data_root, f"amazon-{args.dataset}")
    logging.info(f"Loading temporal fraud data from {data_dir}")

    # Reuse the unified loader (all_nodes.csv / all_edges.csv + node masks)
    data = load_data(data_dir, val_size=0.2, test_size=0.2, args=args)
    data = data.to(device)

    num_all_nodes = data.num_nodes

    # Build TPNet-style neighbor sampler and random features on the full temporal graph
    e_id = torch.arange(data.src.size(0), device=data.src.device, dtype=torch.long) + 1
    train_data = TemporalData(
        src=data.src,
        dst=data.dst,
        t=data.t,
        e_id=e_id,
    )
    neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
        train_data, num_all_nodes, args.seed
    )
    node_features, edge_features = build_tpnet_features(
        num_all_nodes, edge_num, args.embedding_dim, args.seed
    )
    # Edge-level message features aligned with TemporalData event order.
    # `e_id` runs from 1..edge_num, so we drop the padding row at index 0.
    train_data.msg = torch.from_numpy(edge_features[1 : edge_num + 1]).to(device)
    # Optionally initialize user node features from SBERT review sequences.
    if args.use_sequence_embedding:
        init_tpnet_user_features_from_reviews(data, node_features, data_dir, args)

    encoder = None
    random_projections = None

    if args.model == "TPNet":
        # Random projection module for TPNet
        beginning_time = float(train_data.t[0].item())
        random_projections = RandomProjectionModule(
            node_num=num_all_nodes + 1,
            edge_num=edge_num,
            dim_factor=args.rp_dim_factor,
            num_layer=args.rp_num_layer,
            time_decay_weight=args.rp_time_decay_weight,
            device=str(device),
            use_matrix=args.rp_use_matrix,
            beginning_time=np.float64(beginning_time),
            not_scale=args.rp_not_scale,
            enforce_dim=args.rp_enforce_dim,
        ).to(device)

        encoder = TPNet(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=args.embedding_dim,
            output_dim=args.embedding_dim,
            dropout=0.1,
            random_projections=random_projections,
            num_layers=2,
            num_neighbors=args.num_neighbors,
            device=str(device),
            not_embedding=False,
        ).to(device)
    elif args.model == "DyGFormer":
        encoder = DyGFormer(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=args.embedding_dim,
            output_dim=args.embedding_dim,
            channel_embedding_dim=args.embedding_dim,
            patch_size=1,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            max_input_sequence_length=512,
            device=str(device),
        ).to(device)
    elif args.model == "GraphMixer":
        encoder = GraphMixer(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=args.embedding_dim,
            output_dim=args.embedding_dim,
            # Use the same value for num_tokens and num_neighbors so that
            # the Mixer token dimension matches the neighbor dimension.
            num_tokens=args.num_neighbors,
            num_layers=2,
            token_dim_expansion_factor=0.5,
            channel_dim_expansion_factor=4.0,
            dropout=0.1,
            device=str(device),
        ).to(device)
    elif args.model == "CAWN":
        encoder = CAWN(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=args.embedding_dim,
            output_dim=args.embedding_dim,
            position_feat_dim=args.embedding_dim,
            walk_length=2,
            num_walk_heads=8,
            dropout=0.1,
            device=str(device),
        ).to(device)
    elif args.model == "TGAT":
        encoder = TGAT(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=args.embedding_dim,
            output_dim=args.embedding_dim,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            device=str(device),
        ).to(device)
    elif args.model == "TCL":
        encoder = TCL(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=args.embedding_dim,
            output_dim=args.embedding_dim,
            num_layers=2,
            num_heads=2,
            # Depths must equal num_neighbors + 1 since TCL prepends
            # the target node to the neighbor sequence.
            num_depths=args.num_neighbors + 1,
            dropout=0.1,
            device=str(device),
        ).to(device)
    elif args.model == "NAT":
        encoder = NAT(
            n_feat=node_features,
            e_feat=edge_features,
            output_dim=args.embedding_dim,
            time_dim=2,
            num_neighbors=[1, int(args.num_neighbors)],
            dropout=0.1,
            n_hops=1,
            device=device,
        ).to(device)
        encoder.set_device(device)
        encoder.set_seed(args.seed)
    elif args.model == "CTAN":
        encoder = CTAN(
            num_nodes=num_all_nodes + 1,
            edge_dim=args.embedding_dim,
            memory_dim=args.embedding_dim,
            time_dim=2,
            node_dim=0,
            num_iters=1,
            predict_dst=False,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    nodeclf = MLPClassifier(
        in_channels=args.embedding_dim,
        hidden_channels=args.embedding_dim,
        out_channels=2,
        num_layers=args.clf_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(nodeclf.parameters()), lr=args.lr
    )

    train_loader = TemporalDataLoader(
        train_data, batch_size=args.train_batch_size
    )

    best_val_pr_auc = -float("inf")
    best_test_metrics = None
    best_state = None
    patience_counter = 0

    logging.info(f"Starting {args.model}-based fraud training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        nodeclf.train()
        if args.model == "TPNet" and random_projections is not None:
            random_projections.reset_random_projections()

            # 1) Update temporal random projections over all interactions
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}: updating TPNet state", leave=False):
                batch = batch.to(device)
                src_np = batch.src.cpu().numpy().astype(np.int64) + 1
                dst_np = batch.dst.cpu().numpy().astype(np.int64) + 1
                t_np = batch.t.cpu().numpy().astype(np.float64)
                random_projections.update(src_np, dst_np, t_np)

        # 2) Compute node embeddings (for user nodes) and optimize node classifier
        optimizer.zero_grad()
        z_nodes = compute_node_embeddings_temporal(encoder, data, train_data, device, args)
        logits = nodeclf(z_nodes.to(device))

        num_users = z_nodes.size(0)
        mask = data.train_mask & (data.y != -1)
        mask_users = mask[:num_users]
        loss = F.cross_entropy(
            logits[mask_users],
            data.y[:num_users][mask_users],
        )
        loss.backward()
        optimizer.step()

        train_loss = float(loss.item())
        logging.info(f"Epoch {epoch:03d} Train Loss: {train_loss:.4f}")

        # Validation and (optional) early stopping
        if epoch % args.eval_every == 0:
            val_metrics = evaluate_split(encoder, nodeclf, data, train_data, device, args, split="val")
            log_dict = {"epoch": epoch, "train_loss": train_loss}
            for k, v in val_metrics.items():
                log_dict[f"val_{k}"] = v
            logger.log(log_dict)

            if val_metrics["pr_auc"] > best_val_pr_auc:
                best_val_pr_auc = val_metrics["pr_auc"]
                best_test_metrics = evaluate_split(encoder, nodeclf, data, train_data, device, args, split="test")
                logging.info(
                    f"New best validation PR-AUC: {best_val_pr_auc:.4f}, "
                    f"corresponding test PR-AUC: {best_test_metrics['pr_auc']:.4f}"
                )
                # Save model state at best validation
                best_state = {
                    "encoder": encoder.state_dict(),
                    "nodeclf": nodeclf.state_dict(),
                }
                patience_counter = 0
            elif args.enable_early_stopping:
                patience_counter += 1
                logging.info(
                    f"No improvement in PR-AUC; patience {patience_counter}/{args.patience}"
                )
                if patience_counter >= args.patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}.")
                    break

    elapsed_time = time.time() - start_time
    logging.info(f"Training finished in {elapsed_time:.2f} seconds.")

    # Restore best model state if early stopping was used
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        nodeclf.load_state_dict(best_state["nodeclf"])

    # Final eval
    final_val_metrics = evaluate_split(encoder, nodeclf, data, train_data, device, args, split="val")
    final_test_metrics = evaluate_split(encoder, nodeclf, data, train_data, device, args, split="test")

    logging.info("--- Final Evaluation Results ---")
    logging.info(
        "Final Validation Metrics | "
        + " | ".join(f"{k}: {v:.4f}" for k, v in final_val_metrics.items())
    )
    logging.info(
        "Final Test Metrics       | "
        + " | ".join(f"{k}: {v:.4f}" for k, v in final_test_metrics.items())
    )

    # Log final metrics
    final_results = {}
    for k, v in final_val_metrics.items():
        final_results[f"val_{k}"] = v
    for k, v in final_test_metrics.items():
        final_results[f"test_{k}"] = v
    final_results["train_wall_sec"] = elapsed_time
    logger.log(final_results)
    logger.finish()


def main():
    args = parse_args()
    train_temporal_fraud(args)


if __name__ == "__main__":
    main()