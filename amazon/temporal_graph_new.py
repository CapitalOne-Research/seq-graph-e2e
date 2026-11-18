import os
import sys

# Shared experiment modules live at the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import time
import random
import argparse
import logging
from tqdm import tqdm
import json
import torch
from datetime import datetime
import gc
import copy
import pandas as pd
import numpy as np
from torcheval.metrics import ReciprocalRank, HitRate

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from common.tgb_memory_module import DyRepMemory
from common.tncn import NCNPredictor
from static_graph import LocalLogger

# ---------- NEW IMPORTS: TP-Net components ----------
import sys
import os

# Add repo root (parent of this file's directory) to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from TPNet.random_projection_module import RandomProjectionModule
from TPNet.TPNet import TPNet
from TPNet.dygformer import DyGFormer
from TPNet.graph_mixer import GraphMixer
from TPNet.cawn import CAWN
from TPNet.tgat import TGAT
from TPNet.tcl import TCL
from TPNet.nat import NAT
from TPNet.modules import LinkPredictor_v2
from TPNet.utils import NeighborSampler
# --- CTAN imports ---
from TPNet.ctan.ctdg_models import CTAN



###############################################################################
# Logging configuration
###############################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Link Prediction with Batching on Large Graphs, Early Stopping, and Custom CSV Paths"
    )
    parser.add_argument("--dataset", type=str, choices=['clothing', 'movies', 'electronics', 'musical'],
                        default='musical')
    parser.add_argument('--base_path', type=str, default='.',
                        help="Base path for CSV files.")
    parser.add_argument('--model', type=str,
                        choices=['TGN', 'DyRep', 'TNCN', 'TPNet', 'DyGFormer', 'GraphMixer', 'CAWN', 'TGAT', 'TCL', 'NAT','CTAN'],
                        default='TGN',
                        help="Model type to use.")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help="Dimensionality of input node embeddings.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument('--train_batch_size', type=int, default=2 ** 12,
                        help="Mini-batch size for training.")
    parser.add_argument('--eval_batch_size', type=int, default=2 ** 18,
                        help="Mini-batch size for evaluation and testing.")
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience (in epochs).")
    parser.add_argument('--eval_every', type=int, default=5,
                        help="Evaluation frequency (in epochs).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU ID to use (if CUDA is available).")
    parser.add_argument('--enable_early_stopping', action='store_true',
                        help='Enable early stopping based on validation MRR. (default: False)')
    parser.add_argument('--num_neighbors', type=int, default=10,
                        help="Number of neighbors to sample per layer in the neighbor loader.")
    parser.add_argument('--train_neg_sampling_ratio', type=float, default=5,
                        help="Negative sampling ratio for training in the neighbor loader.")
    parser.add_argument('--hits_k', type=str, default="1,3,5,10,20,50,100",
                        help="Comma separated list of k values for computing Hits@k metrics.")
    parser.add_argument('--experiment_name', type=str, default='TG-exp-1',
                        help="Experiment name for saving checkpoints and artifacts.")
    parser.add_argument("--max_events", type=int, default=100,
                        help="Maximum number of events per user (will truncate/pad accordingly)")

    # ---------- NEW ARGS: TP-Net random projection hyperparameters ----------
    parser.add_argument('--rp_dim_factor', type=int, default=2,
                        help="Factor for random projection dimension (dim ≈ rp_dim_factor * log(2 * |E|)).")
    parser.add_argument('--rp_num_layer', type=int, default=2,
                        help="Number of temporal walk layers to maintain.")
    parser.add_argument('--rp_time_decay_weight', type=float, default=1.0,
                        help="Time decay weight λ in exp(-λ Δt).")
    parser.add_argument('--rp_use_matrix', action='store_true',
                        help="If set, explicitly maintain temporal walk matrices (more memory).")
    parser.add_argument('--rp_not_scale', action='store_true',
                        help="Disable scaling of random features (use raw log-activated features).")
    parser.add_argument('--rp_enforce_dim', type=int, default=-1,
                        help="If > -1, enforce this RP dimension instead of log rule.")
    parser.add_argument('--rp_not_encode', action='store_true',
                        help="If set, ignore base embeddings and use only random-walk features in TP-Net decoder.")
    parser.add_argument('--dygformer_channel_dim', type=int, default=64,
                        help="Channel embedding dimension for DyGFormer.")
    parser.add_argument('--dygformer_patch_size', type=int, default=4,
                        help="Patch size for DyGFormer.")
    parser.add_argument('--dygformer_layers', type=int, default=2,
                        help="Number of transformer layers in DyGFormer.")
    parser.add_argument('--dygformer_heads', type=int, default=2,
                        help="Number of attention heads in DyGFormer.")
    parser.add_argument('--dygformer_dropout', type=float, default=0.1,
                        help="Dropout rate for DyGFormer.")
    parser.add_argument('--dygformer_max_seq_len', type=int, default=512,
                        help="Maximum sequence length for DyGFormer input.")
    parser.add_argument('--gm_num_tokens', type=int, default=20,
                        help="Number of tokens (neighbors) for GraphMixer's mixer stack.")
    parser.add_argument('--gm_num_layers', type=int, default=2,
                        help="Number of MLP-Mixer layers in GraphMixer.")
    parser.add_argument('--gm_token_dim_factor', type=float, default=0.5,
                        help="Token dimension expansion factor for GraphMixer mixers.")
    parser.add_argument('--gm_channel_dim_factor', type=float, default=4.0,
                        help="Channel dimension expansion factor for GraphMixer mixers.")
    parser.add_argument('--gm_dropout', type=float, default=0.1,
                        help="Dropout rate for GraphMixer mixers.")
    parser.add_argument('--gm_num_neighbors', type=int, default=20,
                        help="Number of neighbors sampled per node for GraphMixer.")
    parser.add_argument('--gm_time_gap', type=int, default=2000,
                        help="Time gap (window length) for GraphMixer's node encoder.")
    parser.add_argument('--cawn_num_neighbors', type=int, default=20,
                        help="Number of neighbors sampled per node for CAWN random walks.")
    parser.add_argument('--cawn_walk_length', type=int, default=2,
                        help="Length of anonymous random walks in CAWN.")
    parser.add_argument('--cawn_num_walk_heads', type=int, default=8,
                        help="Number of attention heads for CAWN walk aggregation.")
    parser.add_argument('--cawn_position_dim', type=int, default=32,
                        help="Dimensionality of CAWN position encoder features.")
    parser.add_argument('--cawn_dropout', type=float, default=0.1,
                        help="Dropout for CAWN walk encoder/transformer.")
    parser.add_argument('--tgat_num_layers', type=int, default=2,
                        help="Number of temporal convolution layers in TGAT.")
    parser.add_argument('--tgat_num_heads', type=int, default=2,
                        help="Number of attention heads in TGAT.")
    parser.add_argument('--tgat_dropout', type=float, default=0.1,
                        help="Dropout rate in TGAT.")
    parser.add_argument('--tgat_num_neighbors', type=int, default=20,
                        help="Number of neighbors sampled per layer in TGAT.")
    parser.add_argument('--tcl_num_layers', type=int, default=2,
                        help="Number of transformer layers in TCL.")
    parser.add_argument('--tcl_num_heads', type=int, default=2,
                        help="Number of attention heads in TCL.")
    parser.add_argument('--tcl_num_depths', type=int, default=21,
                        help="Number of depths (neighbors + 1) TCL attends over.")
    parser.add_argument('--tcl_dropout', type=float, default=0.1,
                        help="Dropout for TCL transformers.")
    parser.add_argument('--tcl_num_neighbors', type=int, default=20,
                        help="Number of neighbors sampled per node for TCL.")
    parser.add_argument('--nat_time_dim', type=int, default=2,
                        help="Time encoding dimension for NAT.")
    parser.add_argument('--nat_num_neighbors', type=str, default="1,32",
                        help="Comma separated list of neighbor counts per hop for NAT (e.g., '1,32').")
    parser.add_argument('--nat_n_hops', type=int, default=1,
                        help="Number of hops for NAT neighborhood cache.")
    parser.add_argument('--nat_ngh_dim', type=int, default=4,
                        help="Hidden dimension for NAT neighbor encoder.")
    parser.add_argument('--nat_dropout', type=float, default=0.1,
                        help="Dropout rate for NAT.")
    parser.add_argument('--ctan_gnn_act', type=str, default='relu',
                        help="CTAN GNN nonlinearity: relu | tanh | identity.")
    parser.add_argument('--ctan_gamma', type=float, default=1.0,
                        help="CTAN temporal decay scaling (gamma).")
    parser.add_argument('--ctan_readout_hidden', type=int, default=0,
                        help="Hidden size for CTAN readout MLP (0 = linear).")
    parser.add_argument('--ctan_predict_dst', action='store_true',
                        help="If set, use unary readout p(dst | z_dst). Default is pairwise readout p(edge|z_src,z_dst).")
    parser.add_argument('--ctan_conv_type', type=str, default='graph_attention',
                        help="CTAN conv type: graph_attention | ctan | jodie (if supported).")
    return parser.parse_args()

###############################################################################
# Set Random Seed for Reproducibility
###############################################################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################################
# Data Loading
###############################################################################
def load_data(base_path, embedding_dim, max_events):
    """
    Load edge lists from CSV files with a 'relational_' prefix and create a PyG TemporalData object.

    The files are expected to be:
      - relational_train.csv
      - relational_val.csv
      - relational_test.csv
      - relational_val_negative_sample_100.csv
      - relational_test_negative_sample_100.csv
      - personal_observed.csv
    """
    logging.info("Preparing dataset...")
    logging.info(f"base path: {base_path}")

    train_path = os.path.join(base_path, "relational_train.csv")
    val_path = os.path.join(base_path, "relational_val.csv")
    test_path = os.path.join(base_path, "relational_test.csv")
    val_neg_path = os.path.join(base_path, "relational_val_negative_sample_100.csv")
    test_neg_path = os.path.join(base_path, "relational_test_negative_sample_100.csv")
    personal_path = os.path.join(base_path, "personal_observed.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    val_neg_df = pd.read_csv(val_neg_path)
    test_neg_df = pd.read_csv(test_neg_path)
    personal_df = pd.read_csv(personal_path)

    if max_events > -1:
        personal_df = personal_df.sort_values("timestamp").groupby("uid", group_keys=False) \
            .tail(max_events).sort_values(["uid", "timestamp"])

    num_relational_nodes = max(
        train_df['uid'].max(), train_df['other_uid'].max(),
        val_df['uid'].max(), val_df['other_uid'].max(),
        test_df['uid'].max(), test_df['other_uid'].max()
    ) + 1

    checkin_nodes = personal_df['event'].unique()
    num_personal_nodes = len(checkin_nodes)
    num_all_nodes = num_relational_nodes + num_personal_nodes

    # Convert personal event ids to numeric codes
    codes, _ = pd.factorize(personal_df['event'])
    codes = codes + num_relational_nodes

    # Train data
    src_r = torch.tensor(train_df['uid'].values, dtype=torch.long)
    dst_r = torch.tensor(train_df['other_uid'].values, dtype=torch.long)
    t_r = torch.tensor(train_df['timestamp'].values, dtype=torch.long)

    src_p = torch.tensor(personal_df['uid'].values, dtype=torch.long)
    dst_p = torch.tensor(codes, dtype=torch.long)
    t_p = torch.tensor(personal_df['timestamp'].values, dtype=torch.long)

    # combine and reorder
    t = torch.cat([t_p, t_r], dim=0)
    src = torch.cat([src_p, src_r], dim=0)
    dst = torch.cat([dst_p, dst_r], dim=0)

    # reorder by time
    indices = torch.argsort(t)
    src = src[indices]
    dst = dst[indices]
    t = t[indices]

    train_data = TemporalData(src=src, dst=dst, t=t)

    # Validation data (positive and negative)
    src_val_pos = torch.tensor(val_df['uid'].values, dtype=torch.long)
    dst_val_pos = torch.tensor(val_df['other_uid'].values, dtype=torch.long)
    t_val_pos = torch.ones(len(src_val_pos), dtype=torch.long) * (t_p.max() + 1000)

    src_val_neg = torch.tensor(val_neg_df['uid'].values, dtype=torch.long)
    dst_val_neg = torch.tensor(val_neg_df['other_uid'].values, dtype=torch.long)
    t_val_neg = torch.ones(len(src_val_neg), dtype=torch.long) * (t_p.max() + 1000)

    val_data_pos = TemporalData(src=src_val_pos, dst=dst_val_pos, t=t_val_pos)
    val_data_neg = TemporalData(src=src_val_neg, dst=dst_val_neg, t=t_val_neg)

    # Test data (positive and negative)
    src_test_pos = torch.tensor(test_df['uid'].values, dtype=torch.long)
    dst_test_pos = torch.tensor(test_df['other_uid'].values, dtype=torch.long)
    t_test_pos = torch.ones(len(src_test_pos), dtype=torch.long) * (t_p.max() + 1000)

    src_test_neg = torch.tensor(test_neg_df['uid'].values, dtype=torch.long)
    dst_test_neg = torch.tensor(test_neg_df['other_uid'].values, dtype=torch.long)
    t_test_neg = torch.ones(len(src_test_neg), dtype=torch.long) * (t_p.max() + 1000)

    test_data_pos = TemporalData(src=src_test_pos, dst=dst_test_pos, t=t_test_pos)
    test_data_neg = TemporalData(src=src_test_neg, dst=dst_test_neg, t=t_test_neg)

    logging.info('>> Data stats')
    logging.info(f'>> Train data: src.shape {train_data.src.shape}, num nodes: {train_data.num_nodes}')
    logging.info(f'>> Val data pos: src.shape {val_data_pos.src.shape}, num nodes: {val_data_pos.num_nodes}')
    logging.info(f'>> Val data neg: src.shape {val_data_neg.src.shape}, num nodes: {val_data_neg.num_nodes}')
    logging.info(f'>> Test data pos: src.shape {test_data_pos.src.shape}, num nodes: {test_data_pos.num_nodes}')
    logging.info(f'>> Test data neg: src.shape {test_data_neg.src.shape}, num nodes: {test_data_neg.num_nodes}')

    data = {
        'train_data': train_data,
        'val_data_pos': val_data_pos,
        'val_data_neg': val_data_neg,
        'test_data_pos': test_data_pos,
        'test_data_neg': test_data_neg,
        'num_all_nodes': num_all_nodes
    }
    return data

def build_tpnet_neighbor_sampler(train_data, num_nodes, seed, sample_strategy='uniform', time_scaling=0.0):
    """
    Construct TPNet's NeighborSampler from chronological training edges.
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
    """
    rng = np.random.default_rng(seed)
    node_features = rng.standard_normal((num_nodes + 1, feat_dim)).astype(np.float32)
    edge_features = rng.standard_normal((edge_num + 1, feat_dim)).astype(np.float32)
    node_features[0] = 0.0
    edge_features[0] = 0.0
    return node_features, edge_features

###############################################################################
# Model Definitions
###############################################################################
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class LinkPredictor(torch.nn.Module):
    """
    Simple MLP link predictor (used for TGN / DyRep / baseline).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

###############################################################################
# Evaluation Functions using torcheval
###############################################################################
def evaluate(candidates, target, k_list):
    """
    Compute retrieval metrics using torcheval.metrics.
    """
    if candidates is None or not torch.isfinite(candidates).all():
        nan_metrics = {"MRR": float("nan")}
        nan_metrics.update({f"Hits@{k}": float("nan") for k in k_list})
        return nan_metrics
    mrr_metric = ReciprocalRank()
    hits_metrics = {k: HitRate(k=k) for k in k_list}

    mrr_metric.update(candidates, target)
    for k, hits_metric in hits_metrics.items():
        hits_metric.update(candidates, target)

    mrr = mrr_metric.compute().mean().item()
    hits_results = {f"Hits@{k}": hits_metrics[k].compute().mean().item() for k in k_list}
    return {"MRR": mrr, **hits_results}
def test(model, data, eval_batch_size, device, hits_k, split="test"):
    """
    Evaluate model performance on a given split using torcheval.metrics.
    Handles three families:
      (A) Temporal-encoder family (TPNet, DyGFormer, CAWN, TGAT, TCL, NAT)
      (B) CTAN (owns its memory/conv/predictor internally)
      (C) PyG-memory baselines (TGN, DyRep, TNCN)
    """
    # --- Extract modules/flags from model dict ---
    embedding_module   = model.get('embedding_module')
    memory             = model.get('memory')
    gnn                = model.get('gnn')
    linkpred           = model.get('linkpred')
    neighbor_loader    = model.get('neighbor_loader')
    assoc              = model.get('assoc')
    assoc2             = model.get('assoc2')
    temporal_encoder   = model.get('temporal_encoder')
    temporal_linkpred  = model.get('temporal_linkpred')
    temporal_type      = model.get('temporal_encoder_type')
    temporal_kwargs    = model.get('temporal_encoder_kwargs', {})
    ctan               = model.get('ctan')  # present iff --model CTAN

    # --- Eval mode selection ---
    if temporal_encoder is None:
        # Baseline family
        if ctan is not None:
            ctan.eval()
            if embedding_module is not None:
                embedding_module.eval()
        else:
            # TGN/DyRep/TNCN path
            if embedding_module is not None: embedding_module.eval()
            if memory is not None:           memory.eval()
            if gnn is not None:              gnn.eval()
            if linkpred is not None:         linkpred.eval()
    else:
        # Temporal-encoder family
        temporal_encoder.eval()
        temporal_linkpred.eval()

    pos_scores_list, neg_scores_list = [], []
    train_data = data['train_data']

    if split == "val":
        desc_prefix = "Validation"
        pos_data, neg_data = data['val_data_pos'], data['val_data_neg']
        logging.info("Starting evaluation...")
    else:
        desc_prefix = "Testing"
        pos_data, neg_data = data['test_data_pos'], data['test_data_neg']
        logging.info("Starting testing...")

    pos_loader = TemporalDataLoader(pos_data, batch_size=eval_batch_size, neg_sampling_ratio=0)
    neg_loader = TemporalDataLoader(neg_data, batch_size=eval_batch_size, neg_sampling_ratio=0)

    with torch.no_grad():
        # ================================
        # (A) Temporal-encoder family
        # ================================
        if temporal_encoder is not None:
            # POS
            for batch in tqdm(pos_loader, desc=f"{desc_prefix} Pos", leave=False):
                batch = batch.to(device)
                src_base = batch.src.cpu().numpy()
                dst_base = batch.dst.cpu().numpy()
                src_np   = src_base + 1
                dst_np   = dst_base + 1
                times_np = batch.t.cpu().numpy().astype(np.float64)

                if temporal_type == 'NAT':
                    edge_ids = np.zeros_like(src_base)
                    emb = temporal_encoder.compute_edge_temporal_embeddings(
                        src_base, dst_base, times_np, edge_ids=edge_ids, edges_are_positive=False
                    )
                    scores = temporal_linkpred(emb)
                else:
                    src_emb, dst_emb = temporal_encoder.compute_src_dst_node_temporal_embeddings(
                        src_np, dst_np, times_np, **temporal_kwargs
                    )
                    scores = temporal_linkpred(torch.cat([src_emb, dst_emb], dim=-1))
                pos_scores_list.append(scores.cpu())

            # NEG
            for batch in tqdm(neg_loader, desc=f"{desc_prefix} Neg", leave=False):
                batch = batch.to(device)
                src_base = batch.src.cpu().numpy()
                dst_base = batch.dst.cpu().numpy()
                src_np   = src_base + 1
                dst_np   = dst_base + 1
                times_np = batch.t.cpu().numpy().astype(np.float64)

                if temporal_type == 'NAT':
                    edge_ids = np.zeros_like(src_base)
                    emb = temporal_encoder.compute_edge_temporal_embeddings(
                        src_base, dst_base, times_np, edge_ids=edge_ids, edges_are_positive=False
                    )
                    scores = temporal_linkpred(emb)
                else:
                    src_emb, dst_emb = temporal_encoder.compute_src_dst_node_temporal_embeddings(
                        src_np, dst_np, times_np, **temporal_kwargs
                    )
                    scores = temporal_linkpred(torch.cat([src_emb, dst_emb], dim=-1))
                neg_scores_list.append(scores.cpu())

        # ================================
        # (B) CTAN
        # ================================
        elif ctan is not None:
            # POS
            for batch in tqdm(pos_loader, desc=f"{desc_prefix} Pos", leave=False):
                batch = batch.to(device)

                # Sample neighborhood and build index maps
                n_id, edge_index, e_id = neighbor_loader(batch.n_id)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                # Build message features from learned node embeddings
                neigh_nid = torch.unique(torch.cat([batch.src, batch.dst, edge_index[0], edge_index[1]]))
                assoc2[neigh_nid] = torch.arange(neigh_nid.size(0), device=device)
                x = embedding_module(neigh_nid)
                msg = torch.cat([x[assoc2[edge_index[0]]], x[assoc2[edge_index[1]]]], dim=1)

                # Timestamps for sampled edges
                t_e = train_data.t[e_id.cpu()].to(device)

                # CTAN forward returns (pos_out, neg_out, m_src, m_dst)
                pos_out, _, _, _ = ctan.forward(
                    batch=batch, n_id=n_id, msg=msg, t=t_e, edge_index=edge_index, id_mapper=assoc
                )
                pos_scores_list.append(pos_out.cpu())

            # NEG
            for batch in tqdm(neg_loader, desc=f"{desc_prefix} Neg", leave=False):
                batch = batch.to(device)

                n_id, edge_index, e_id = neighbor_loader(batch.n_id)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                neigh_nid = torch.unique(torch.cat([batch.src, batch.dst, edge_index[0], edge_index[1]]))
                assoc2[neigh_nid] = torch.arange(neigh_nid.size(0), device=device)
                x = embedding_module(neigh_nid)
                msg = torch.cat([x[assoc2[edge_index[0]]], x[assoc2[edge_index[1]]]], dim=1)

                t_e = train_data.t[e_id.cpu()].to(device)

                neg_out_like_pos, _, _, _ = ctan.forward(
                    batch=batch, n_id=n_id, msg=msg, t=t_e, edge_index=edge_index, id_mapper=assoc
                )
                # Here, the loader already provides "negative edges"; take the primary output.
                neg_scores_list.append(neg_out_like_pos.cpu())

        # ================================
        # (C) PyG-memory baselines (TGN/DyRep/TNCN)
        # ================================
        else:
            # POS
            for batch in tqdm(pos_loader, desc=f"{desc_prefix} Pos", leave=False):
                batch = batch.to(device)
                n_id, edge_index, e_id = neighbor_loader(batch.n_id)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                neigh_nid = torch.unique(torch.cat([batch.src, batch.dst, edge_index[0]]))
                assoc2[neigh_nid] = torch.arange(neigh_nid.size(0), device=device)

                x = embedding_module(neigh_nid)
                msg = torch.cat([x[assoc2[edge_index[0]]], x[assoc2[edge_index[1]]]], dim=1)

                z, last_update = memory(n_id)
                z = gnn(z, last_update, edge_index, train_data.t[e_id.cpu()].to(device), msg)

                if 'TNCN' in globals() and 'NCN_MODE' in globals() and 'CN_TIME_DECAY' in globals() and getattr(args, 'model', None) == 'TNCN':
                    time_info = (last_update, batch.t)
                    scores = linkpred(
                        z, edge_index, torch.stack([assoc[batch.src], assoc[batch.dst]]),
                        NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info
                    )
                else:
                    scores = linkpred(z[assoc[batch.src]], z[assoc[batch.dst]])
                pos_scores_list.append(scores.cpu())

            # NEG
            for batch in tqdm(neg_loader, desc=f"{desc_prefix} Neg", leave=False):
                batch = batch.to(device)
                n_id, edge_index, e_id = neighbor_loader(batch.n_id)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                neigh_nid = torch.unique(torch.cat([batch.src, batch.dst, edge_index[0]]))
                assoc2[neigh_nid] = torch.arange(neigh_nid.size(0), device=device)

                x = embedding_module(neigh_nid)
                msg = torch.cat([x[assoc2[edge_index[0]]], x[assoc2[edge_index[1]]]], dim=1)

                z, last_update = memory(n_id)
                z = gnn(z, last_update, edge_index, train_data.t[e_id.cpu()].to(device), msg)

                if 'TNCN' in globals() and 'NCN_MODE' in globals() and 'CN_TIME_DECAY' in globals() and getattr(args, 'model', None) == 'TNCN':
                    time_info = (last_update, batch.t)
                    scores = linkpred(
                        z, edge_index, torch.stack([assoc[batch.src], assoc[batch.dst]]),
                        NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info
                    )
                else:
                    scores = linkpred(z[assoc[batch.src]], z[assoc[batch.dst]])
                neg_scores_list.append(scores.cpu())

    # --- Stack to candidates/targets (same layout as before) ---
    pos_scores = torch.cat(pos_scores_list, dim=0)
    neg_scores = torch.cat(neg_scores_list, dim=0)

    num_samples = pos_scores.shape[0]
    num_neg = neg_scores.shape[0] // num_samples
    neg_scores = neg_scores.reshape(num_samples, num_neg)

    candidates = torch.cat([pos_scores, neg_scores], dim=1)
    target = torch.zeros(num_samples, dtype=torch.long)  # positive sample at index 0

    metrics = evaluate(candidates, target, k_list=hits_k)
    return metrics

def train(model, data, optimizer, criterion, epochs, train_batch_size, eval_batch_size,
          device, patience, eval_every, enable_early_stopping, train_neg_sampling_ratio, hits_k):
    """
    Train the model with batching and (optionally) early stopping.
    """
    train_data = data['train_data']
    train_loader = TemporalDataLoader(
        train_data, batch_size=train_batch_size,
        neg_sampling_ratio=train_neg_sampling_ratio
    )

    embedding_module = model.get('embedding_module')
    memory = model.get('memory')
    gnn = model.get('gnn')
    linkpred = model.get('linkpred')
    neighbor_loader = model.get('neighbor_loader')
    assoc = model.get('assoc')
    assoc2 = model.get('assoc2')
    random_projections = model.get('random_projections')
    temporal_encoder = model.get('temporal_encoder')
    temporal_linkpred = model.get('temporal_linkpred')
    temporal_encoder_type = model.get('temporal_encoder_type')
    temporal_encoder_kwargs = model.get('temporal_encoder_kwargs', {})

    if temporal_encoder is None:
        if embedding_module is not None:
            embedding_module.train()
        if memory is not None:
            memory.train()
        if gnn is not None:
            gnn.train()
        if linkpred is not None:
            linkpred.train()
    else:
        temporal_encoder.train()
        temporal_linkpred.train()

    best_val_mrr = -float('inf')
    patience_counter = 0
    best_state = None

    logging.info("Starting training...")

    last_epoch_time = None
    last_epoch_timestamp = None

    for epoch in range(1, epochs + 1):
        # record start time and timestamp for this epoch
        epoch_start_time = time.time()
        epoch_timestamp = datetime.now().isoformat()

        if memory is not None:
            memory.reset_state()
        if neighbor_loader is not None:
            neighbor_loader.reset_state()
        if temporal_encoder_type == 'TPNet' and random_projections is not None:
            random_projections.reset_random_projections()

        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            if temporal_encoder is not None:
                ratio = max(1, int(train_neg_sampling_ratio))
                pos_src_base = batch.src.cpu().numpy()
                pos_dst_base = batch.dst.cpu().numpy()
                pos_src_np = pos_src_base + 1
                pos_dst_np = pos_dst_base + 1
                times_np = batch.t.cpu().numpy().astype(np.float64)

                if temporal_encoder_type == 'TPNet' and random_projections is not None:
                    random_projections.update(pos_src_np, pos_dst_np, times_np)

                if temporal_encoder_type == 'NAT':
                    edge_ids = np.zeros_like(pos_src_base)
                    pos_emb = temporal_encoder.compute_edge_temporal_embeddings(
                        pos_src_base, pos_dst_base, times_np, edge_ids=edge_ids,
                        edges_are_positive=True
                    )
                    pos_logits = temporal_linkpred(pos_emb)

                    neg_src_base = batch.src.repeat_interleave(ratio).cpu().numpy()
                    neg_dst_base = batch.neg_dst.cpu().numpy()
                    neg_src_np = neg_src_base + 1
                    neg_dst_np = neg_dst_base + 1
                    neg_times_np = np.repeat(batch.t.cpu().numpy().astype(np.float64), ratio)
                    neg_edge_ids = np.zeros_like(neg_src_base)
                    neg_emb = temporal_encoder.compute_edge_temporal_embeddings(
                        neg_src_base, neg_dst_base, neg_times_np, edge_ids=neg_edge_ids,
                        edges_are_positive=False
                    )
                    neg_logits = temporal_linkpred(neg_emb)
                else:
                    pos_src_emb, pos_dst_emb = temporal_encoder.compute_src_dst_node_temporal_embeddings(
                        pos_src_np, pos_dst_np, times_np, **temporal_encoder_kwargs
                    )
                    pos_logits = temporal_linkpred(torch.cat([pos_src_emb, pos_dst_emb], dim=-1))

                    neg_src_np = np.repeat(pos_src_np, ratio)
                    neg_dst_np = batch.neg_dst.cpu().numpy() + 1
                    neg_times_np = np.repeat(times_np, ratio)
                    neg_src_emb, neg_dst_emb = temporal_encoder.compute_src_dst_node_temporal_embeddings(
                        neg_src_np, neg_dst_np, neg_times_np, **temporal_encoder_kwargs
                    )
                    neg_logits = temporal_linkpred(torch.cat([neg_src_emb, neg_dst_emb], dim=-1))
            else:
                ratio = max(1, int(train_neg_sampling_ratio))
                n_id, edge_index, e_id = neighbor_loader(batch.n_id)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                neigh_nid = torch.unique(torch.cat([batch.src, batch.dst, edge_index[0], edge_index[1]]))
                assoc2[neigh_nid] = torch.arange(neigh_nid.size(0), device=device)

                # Build per-edge message from learned node embeddings (same as your baseline)
                x = embedding_module(neigh_nid)
                msg = torch.cat([x[assoc2[edge_index[0]]], x[assoc2[edge_index[1]]]], dim=1)

                if model.get('ctan') is not None:
                    # ---- CTAN path ----
                    ctan = model['ctan']
                    # CTAN reads timestamps of edges in the sampled subgraph:
                    # use training graph times at the sampled edge ids
                    pos_logits, neg_logits, m_src, m_dst = ctan.forward(
                        batch=batch,
                        n_id=n_id,
                        msg=msg,
                        t=train_data.t[e_id.cpu()].to(device),
                        edge_index=edge_index,
                        id_mapper=assoc,
                    )
                    # Update memory & neighbor cache (CTAN owns its memory internally).
                    # Use CTAN's update interface so SimpleMemory receives per-node embeddings.
                    msg_gt = torch.cat([x[assoc2[batch.src]], x[assoc2[batch.dst]]], dim=1).detach()
                    ctan.update(batch.src, batch.dst, batch.t, msg_gt, m_src.detach(), m_dst.detach())
                    neighbor_loader.insert(batch.src, batch.dst)
                else:
                    # ---- TGN / DyRep / TNCN path (your existing code) ----
                    z, last_update = memory(n_id)
                    z = gnn(z, last_update, edge_index, train_data.t[e_id.cpu()].to(device), msg)
                    if args.model == 'TNCN':
                        time_info = (last_update, batch.t)
                        pos_logits = linkpred(
                            z, edge_index, torch.stack([assoc[batch.src], assoc[batch.dst]]),
                            NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info
                        )
                        neg_src = batch.src.repeat_interleave(train_neg_sampling_ratio, dim=0)
                        neg_logits = linkpred(
                            z, edge_index,
                            torch.stack([assoc[neg_src], assoc[batch.neg_dst]]),
                            NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info
                        )
                    else:
                        pos_logits = linkpred(z[assoc[batch.src]], z[assoc[batch.dst]])
                        neg_src = batch.src.repeat_interleave(train_neg_sampling_ratio, dim=0)
                        neg_logits = linkpred(z[assoc[neg_src]], z[assoc[batch.neg_dst]])

                    msg_gt = torch.cat([x[assoc2[batch.src]], x[assoc2[batch.dst]]], dim=1).detach()
                    memory.update_state(batch.src, batch.dst, batch.t, msg_gt)
                    neighbor_loader.insert(batch.src, batch.dst)

            loss = criterion(pos_logits, torch.ones_like(pos_logits))
            loss += criterion(neg_logits, torch.zeros_like(neg_logits))

            loss.backward()
            optimizer.step()

            if memory is not None:
                memory.detach()

            total_loss += float(loss) * batch.num_events

        avg_loss = total_loss / train_data.num_events
        epoch_elapsed = time.time() - epoch_start_time
        last_epoch_time = epoch_elapsed
        last_epoch_timestamp = epoch_timestamp
        logging.info(f"Epoch {epoch:03d} Loss: {avg_loss:.4f} | epoch_time: {epoch_elapsed:.2f}s | ts: {epoch_timestamp}")

        if epoch % eval_every == 0:
            metrics = test(model, data, eval_batch_size, device, hits_k, split="val")
            val_mrr = metrics["MRR"]
            log_msg = f"Epoch {epoch:03d} Evaluation | Val MRR: {val_mrr:.4f} | epoch_time: {epoch_elapsed:.2f}s | ts: {epoch_timestamp}"
            for k in hits_k:
                log_msg += f", Hits@{k}: {metrics[f'Hits@{k}']:.4f}"
            logging.info(log_msg)
            log_payload = {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "epoch_time": epoch_elapsed,
                "epoch_timestamp": epoch_timestamp,
                **{f"val_{k.lower().replace('@', '_at_')}": v for k, v in metrics.items()}
            }
            logger.log(log_payload)
            if enable_early_stopping:
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    best_state = {k: copy.deepcopy(v.state_dict())
                                  for k, v in model.items()
                                  if hasattr(v, "load_state_dict")}
                    patience_counter = 0
                    logging.info("New best model found; resetting patience counter.")
                else:
                    patience_counter += 1
                    logging.info(f"No improvement; patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}.")
                    break

    if best_state is not None:
        for k, v in model.items():
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(best_state[k])
    # store last epoch timing in model for access from main
    model['last_epoch_time'] = last_epoch_time
    model['last_epoch_timestamp'] = last_epoch_timestamp
    return model

###############################################################################
# Checkpoint Saving Function
###############################################################################
def save_checkpoint(filepath, model, optimizer, args, elapsed_time, test_metrics):
    """
    Save the model, optimizer, training arguments, elapsed time,
    and test metrics to a checkpoint file.
    """
    checkpoint = {
        'experiment_name': args.experiment_name,
        'model_state_dict': {k: v.state_dict() for k, v in model.items() if hasattr(v, "state_dict")},
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'elapsed_time': elapsed_time,
        'test_metrics': test_metrics,
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved at {filepath}")

###############################################################################
# Main Execution
###############################################################################

###############################################################################
# Model factory
###############################################################################
def build_model(args, data, device):
    """
    Build model dict and return (model, params) where:
      - model: dict with keys like 'embedding_module', 'memory', 'gnn',
               'linkpred', 'neighbor_loader', 'assoc', 'assoc2',
               'temporal_encoder', 'temporal_linkpred',
               'temporal_encoder_type', 'temporal_encoder_kwargs',
               'random_projections' (optional).
      - params: list of parameters to optimize.
    """
    num_all_nodes = data['num_all_nodes']
    train_data = data['train_data']

    embedding_dim = args.embedding_dim
    msg_dim = memory_dim = time_dim = 2 * embedding_dim

    model: dict = {}
    embedding_module = memory = gnn = linkpred = neighbor_loader = assoc = assoc2 = None
    random_projections = None

    # ----- TPNet-family encoders: use TP-Net neighbor sampler + features -----
    if args.model == 'TPNet':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )

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

        tpnet_model = TPNet(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=embedding_dim,
            output_dim=embedding_dim,
            dropout=0.1,
            random_projections=random_projections,
            num_layers=2,
            num_neighbors=args.num_neighbors,
            device=str(device),
            not_embedding=False,
        ).to(device)
        tp_linkpred = LinkPredictor_v2(
            input_dim=2 * embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=1,
        ).to(device)

        model.update({
            'temporal_encoder': tpnet_model,
            'temporal_linkpred': tp_linkpred,
            'temporal_encoder_type': 'TPNet',
            'temporal_encoder_kwargs': {},
            'random_projections': random_projections,
        })

    elif args.model == 'DyGFormer':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )
        dygformer_model = DyGFormer(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=embedding_dim,
            output_dim=embedding_dim,
            channel_embedding_dim=args.dygformer_channel_dim,
            patch_size=args.dygformer_patch_size,
            num_layers=args.dygformer_layers,
            num_heads=args.dygformer_heads,
            dropout=args.dygformer_dropout,
            max_input_sequence_length=args.dygformer_max_seq_len,
            device=str(device),
        ).to(device)
        dy_linkpred = LinkPredictor_v2(
            input_dim=2 * embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=1,
        ).to(device)
        model.update({
            'temporal_encoder': dygformer_model,
            'temporal_linkpred': dy_linkpred,
            'temporal_encoder_type': 'DyGFormer',
            'temporal_encoder_kwargs': {},
        })

    elif args.model == 'CAWN':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )
        cawn_model = CAWN(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=embedding_dim,
            output_dim=embedding_dim,
            position_feat_dim=args.cawn_position_dim,
            walk_length=args.cawn_walk_length,
            num_walk_heads=args.cawn_num_walk_heads,
            dropout=args.cawn_dropout,
            device=str(device),
        ).to(device)
        cawn_linkpred = LinkPredictor_v2(
            input_dim=2 * embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=1,
        ).to(device)
        model.update({
            'temporal_encoder': cawn_model,
            'temporal_linkpred': cawn_linkpred,
            'temporal_encoder_type': 'CAWN',
            'temporal_encoder_kwargs': {'num_neighbors': args.cawn_num_neighbors},
        })

    elif args.model == 'GraphMixer':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )
        graph_mixer_model = GraphMixer(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=embedding_dim,
            output_dim=embedding_dim,
            num_tokens=args.gm_num_tokens,
            num_layers=args.gm_num_layers,
            token_dim_expansion_factor=args.gm_token_dim_factor,
            channel_dim_expansion_factor=args.gm_channel_dim_factor,
            dropout=args.gm_dropout,
            device=str(device),
        ).to(device)
        gm_linkpred = LinkPredictor_v2(
            input_dim=2 * embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=1,
        ).to(device)
        model.update({
            'temporal_encoder': graph_mixer_model,
            'temporal_linkpred': gm_linkpred,
            'temporal_encoder_type': 'GraphMixer',
            'temporal_encoder_kwargs': {
                'num_neighbors': args.gm_num_neighbors,
                'time_gap': args.gm_time_gap,
            },
        })

    elif args.model == 'TGAT':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )
        tgat_model = TGAT(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=embedding_dim,
            output_dim=embedding_dim,
            num_layers=args.tgat_num_layers,
            num_heads=args.tgat_num_heads,
            dropout=args.tgat_dropout,
            device=str(device),
        ).to(device)
        tgat_linkpred = LinkPredictor_v2(
            input_dim=2 * embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=1,
        ).to(device)
        model.update({
            'temporal_encoder': tgat_model,
            'temporal_linkpred': tgat_linkpred,
            'temporal_encoder_type': 'TGAT',
            'temporal_encoder_kwargs': {'num_neighbors': args.tgat_num_neighbors},
        })

    elif args.model == 'NAT':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )
        nat_num_neighbors = [int(x) for x in args.nat_num_neighbors.split(',') if x.strip()]
        nat_model = NAT(
            n_feat=node_features,
            e_feat=edge_features,
            output_dim=embedding_dim,
            time_dim=args.nat_time_dim,
            num_neighbors=nat_num_neighbors,
            dropout=args.nat_dropout,
            n_hops=args.nat_n_hops,
            ngh_dim=args.nat_ngh_dim,
            device=str(device),
        ).to(device)
        nat_model.set_seed(args.seed)
        nat_model.init_ncache()
        nat_linkpred = LinkPredictor_v2(
            input_dim=nat_model.output_dim + nat_model.self_dim * 2,
            hidden_dim=embedding_dim,
            output_dim=1,
        ).to(device)
        model.update({
            'temporal_encoder': nat_model,
            'temporal_linkpred': nat_linkpred,
            'temporal_encoder_type': 'NAT',
            'temporal_encoder_kwargs': {},
        })

    elif args.model == 'TCL':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )
        tcl_model = TCL(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=embedding_dim,
            output_dim=embedding_dim,
            num_layers=args.tcl_num_layers,
            num_heads=args.tcl_num_heads,
            num_depths=args.tcl_num_depths,
            dropout=args.tcl_dropout,
            device=str(device),
        ).to(device)
        tcl_linkpred = LinkPredictor_v2(
            input_dim=2 * embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=1,
        ).to(device)
        model.update({
            'temporal_encoder': tcl_model,
            'temporal_linkpred': tcl_linkpred,
            'temporal_encoder_type': 'TCL',
            'temporal_encoder_kwargs': {'num_neighbors': args.tcl_num_neighbors},
        })
    elif args.model == 'CTAN':
        neighbor_sampler, edge_num = build_tpnet_neighbor_sampler(
            train_data, num_all_nodes, args.seed
        )
        node_features, edge_features = build_tpnet_features(
            num_all_nodes, edge_num, embedding_dim, args.seed
        )

        if train_data.t.numel() > 1:
            dt = (train_data.t[1:].float() - train_data.t[:-1].float())
            mean_dt = float(dt.mean().item())
            std_dt  = float(dt.std(unbiased=False).item()) or 1.0
        else:
            mean_dt, std_dt = 1.0, 1.0

        ctan_model = CTAN(
            num_nodes=num_all_nodes + 1,
            edge_dim=2 * embedding_dim,
            memory_dim=2 * embedding_dim,
            time_dim=2 * embedding_dim,
            node_dim=0,
            num_iters=1,
            gnn_act='relu',
            gnn_act_kwargs={},
            epsilon=1e-6,
            gamma=0.99,
            readout_hidden=embedding_dim,
            final_act=None,
            predict_dst=False,
            conv_type='TransformerConv',
            mean_delta_t=mean_dt,
            std_delta_t=std_dt,
            init_time=float(train_data.t[0].item()),
        ).to(device)

        # Create embedding + neighbor loader for CTAN's message construction
        embedding_module = torch.nn.Embedding(num_all_nodes, embedding_dim).to(device)
        neighbor_loader = LastNeighborLoader(num_all_nodes, size=args.num_neighbors, device=device)
        assoc = torch.empty(num_all_nodes, dtype=torch.long, device=device)
        assoc2 = torch.empty(num_all_nodes, dtype=torch.long, device=device)

        model.update({
            'embedding_module': embedding_module,
            'memory': None,
            'gnn': None,
            'linkpred': None,
            'neighbor_loader': neighbor_loader,
            'assoc': assoc,
            'assoc2': assoc2,
            'ctan': ctan_model,
            'temporal_encoder_type': None,
            'temporal_encoder_kwargs': {},
        })
    # ----- PyG-memory baselines: TGN / DyRep / TNCN -----
    else:
        embedding_module = torch.nn.Embedding(num_all_nodes, embedding_dim).to(device)

        if args.model in ['TGN', 'TNCN']:
            memory = TGNMemory(
                num_all_nodes,
                msg_dim,
                memory_dim,
                time_dim,
                message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
                aggregator_module=LastAggregator(),
            ).to(device)
        else:  # DyRep
            memory = DyRepMemory(
                num_all_nodes,
                msg_dim,
                memory_dim,
                time_dim,
                message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
                aggregator_module=LastAggregator(),
                memory_updater_type='rnn'
            ).to(device)

        gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=memory.time_enc,
        ).to(device)

        if args.model == 'TNCN':
            # expose these as globals so train() can access them
            global NCN_MODE, CN_TIME_DECAY
            NCN_MODE = 1
            CN_TIME_DECAY = False
            linkpred = NCNPredictor(
                in_channels=embedding_dim,
                hidden_channels=embedding_dim,
                out_channels=1,
                NCN_mode=NCN_MODE
            ).to(device)
        else:
            linkpred = LinkPredictor(in_channels=embedding_dim).to(device)

        neighbor_loader = LastNeighborLoader(num_all_nodes, size=args.num_neighbors, device=device)
        assoc = torch.empty(num_all_nodes, dtype=torch.long, device=device)
        assoc2 = torch.empty(num_all_nodes, dtype=torch.long, device=device)

        model.update({
            'embedding_module': embedding_module,
            'memory': memory,
            'gnn': gnn,
            'linkpred': linkpred,
            'neighbor_loader': neighbor_loader,
            'assoc': assoc,
            'assoc2': assoc2,
            'temporal_encoder_type': None,
            'temporal_encoder_kwargs': {},
        })

    # ----- Collect parameters for optimizer -----
    params = []

    # Baseline modules
    for module in [embedding_module, memory, gnn, linkpred]:
        if module is not None:
            params.extend(list(module.parameters()))

    # Temporal encoders + their link predictors
    temporal_encoder = model.get('temporal_encoder')
    temporal_linkpred = model.get('temporal_linkpred')

    if temporal_encoder is not None:
        params.extend(list(temporal_encoder.parameters()))
    if temporal_linkpred is not None:
        params.extend(list(temporal_linkpred.parameters()))

    ctan_model = model.get('ctan')
    if ctan_model is not None:
        params.extend(list(ctan_model.parameters()))

    # Random projection module (TPNet)
    if random_projections is not None:
        params.extend(list(random_projections.parameters()))

    return model, params


###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    args = parse_args()
    args.hits_k = [int(x) for x in args.hits_k.split(",")]
    # args.base_path = (
    #     f"amazon-{args.dataset}/coreview_prediction/"
    # )
    args.base_path = (
        f"../data/tasks/"
        f"amazon-{args.dataset}/coreview_prediction/"
    )

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    project_name = f"temporal-results-{args.dataset}"
    run_name = (
        f"{args.model.lower()}_lr{args.lr}_bs{args.train_batch_size}"
        f"_nngb{args.num_neighbors}_epochs{args.epochs}_seed{args.seed}"
    )
    logger = LocalLogger(project_name=project_name, run_name=run_name)

    if logger.skip:
        raise SystemExit(0)

    data = load_data(args.base_path, args.embedding_dim, args.max_events)

    model, params = build_model(args, data, device)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    start_time = time.time()
    model = train(
        model, data, optimizer, criterion, args.epochs,
        args.train_batch_size, args.eval_batch_size, device,
        patience=args.patience, eval_every=args.eval_every,
        enable_early_stopping=args.enable_early_stopping,
        train_neg_sampling_ratio=args.train_neg_sampling_ratio,
        hits_k=args.hits_k,
    )
    elapsed_time = time.time() - start_time

    test_metrics = test(model, data, args.eval_batch_size, device, args.hits_k, split="test")
    log_msg = f"{args.model}: MRR: {test_metrics['MRR']:.4f}"
    for k in args.hits_k:
        log_msg += f", Hits@{k}: {test_metrics[f'Hits@{k}']:.4f}"
    log_msg += f", Time: {elapsed_time:.2f}s"
    logging.info(log_msg)

    final_log_payload = {
        "summary": True,
        "selection": "validation_mrr" if args.enable_early_stopping else "final_epoch",
        "elapsed_time": elapsed_time,
        "last_epoch_time": model.get('last_epoch_time', None),
        "last_epoch_timestamp": model.get('last_epoch_timestamp', None),
        **{f"test_{k.lower().replace('@', '_at_')}": v for k, v in test_metrics.items()},
    }
    logger.log(final_log_payload)
    logger.finish()

    os.makedirs("stored", exist_ok=True)
    checkpoint_filename = f"stored/{args.experiment_name}_checkpoint.pt"
    save_checkpoint(checkpoint_filename, model, optimizer, args, elapsed_time, test_metrics)
