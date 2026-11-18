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

import torch
import gc
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
import torch_geometric.transforms as T
from common.tgb_memory_module import DyRepMemory
from review_embedding import *
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
import torch
import torch.nn.functional as F
import numpy as np



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
    parser.add_argument("--dataset", type=str, choices=['musical', 'movies', 'clothing', 'electronics', 'home-kitchen', 'office-products'], default='musical')
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"),
                        help="dir with amazon-<dataset>/ (all_nodes.csv, all_edges.csv, splits/)")
    parser.add_argument("--log_dir", type=str, default="logs_delta",
                        help="root dir for LocalLogger JSONs; use a fresh dir for the canonical rerun")
    parser.add_argument('--base_path', type=str, default='.',
                        help="Base path for CSV files (nodes.csv, edges.csv).")
    parser.add_argument('--model', type=str, choices=['TGN', 'DyRep','TNCN'], default='TGN',
                        help="Model type to use.")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help="Dimensionality of input node embeddings.")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Maximum number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument('--train_batch_size', type=int, default=2**12,
                        help="Mini-batch size for training.")
    parser.add_argument('--clf_layers', type=int, default=2, help="Number of layers in the MLP classifier.")
    parser.add_argument('--eval_batch_size', type=int, default=2**18,
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
    parser.add_argument('--experiment_name', type=str, default='TG-exp-1',
                        help="Experiment name for saving checkpoints and artifacts.")
    parser.add_argument("--max_events", type=int, default=50,
                        help="Maximum number of events per user (will truncate/pad accordingly)")
    parser.add_argument(
        '--use_sequence_embedding',
        action='store_true',
        help="Initialize user node embeddings from SBERT review sequences.",
    )

    # Data split management (consistent with other scripts)
    parser.add_argument('--force_new_splits', action='store_true', default=False, 
                        help="Force creation of new data splits even if existing ones are found")
    
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
# Data Split Management Functions
###############################################################################
def save_tg_data_splits(splits, dataset_name, seed, split_dir):
    """Save data splits to a file for reproducibility using same format as other scripts."""
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, f'{dataset_name}_splits_seed{seed}.pt')
    
    torch.save(splits, split_file)
    print(f"Data splits saved to {split_file}")
    return split_file

def load_tg_data_splits(dataset_name, seed, split_dir, device):
    """Load data splits from a file using same format as other scripts."""
    split_file = os.path.join(split_dir, f'{dataset_name}_splits_seed{seed}.pt')
    
    if os.path.exists(split_file):
        splits = torch.load(split_file, map_location=device)
        print(f"Data splits loaded from {split_file}")
        return splits['train_mask'], splits['val_mask'], splits['test_mask']
    else:
        print(f"No split file found at {split_file}")
        return None, None, None

###############################################################################
# Data Loading
###############################################################################


def load_data(base_path, val_size=0.2, test_size=0.2, args=None):
    """
    Loads node and edge data, processes them into a temporal event stream,
    and splits the events chronologically into training, validation, and test sets.

    Returns:
        dict: A dictionary containing 'train_data', 'val_data', 'test_data'
              as TemporalData objects, and 'num_all_nodes'.
    """
    logging.info("Preparing dataset from static graph files...")
    logging.info(f"Base path: {base_path}")

    nodes_path = os.path.join(base_path, "all_nodes.csv")
    edges_path = os.path.join(base_path, "all_edges.csv")
    
    # --- 1. Read and Map Raw Data ---
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    all_user_ids = sorted(pd.concat([nodes_df['uid'], edges_df['uid'], edges_df['other_uid']]).unique())
    all_event_ids = sorted(nodes_df['event'].unique())

    # Create mappings from original IDs to new, zero-indexed integers
    # This is crucial for PyG models
    node_mapping = {uid: i for i, uid in enumerate(all_user_ids)}
    event_mapping = {evt: i + len(all_user_ids) for i, evt in enumerate(all_event_ids)}

    num_users = len(all_user_ids)
    num_events_nodes = len(all_event_ids)
    num_all_nodes = num_users + num_events_nodes
        
    # --- 2. Process and Combine All Temporal Events ---
    # Process relational edges (user-user)
    src_rel = edges_df['uid'].map(node_mapping)
    dst_rel = edges_df['other_uid'].map(node_mapping)
    t_rel = torch.from_numpy(edges_df['timestamp'].values).long()

    # Process personal edges (user-event)
    src_pers = nodes_df['uid'].map(node_mapping)
    dst_pers = nodes_df['event'].map(event_mapping)
    t_pers = torch.from_numpy(nodes_df['timestamp'].values).long()

    # Combine all events into a single stream
    src = torch.cat([torch.from_numpy(src_rel.values), torch.from_numpy(src_pers.values)])
    dst = torch.cat([torch.from_numpy(dst_rel.values), torch.from_numpy(dst_pers.values)])
    t = torch.cat([t_rel, t_pers])

    # Sort events chronologically by timestamp
    sorted_indices = torch.argsort(t)
    src = src[sorted_indices]
    dst = dst[sorted_indices]
    t = t[sorted_indices]

    # --- 3. Create the Global Labels Tensor (y) ---
    user_labels_df = nodes_df[['uid', 'fraud_label']].drop_duplicates().set_index('uid')
    y = torch.full((num_all_nodes,), -1, dtype=torch.long) # -1 for nodes with no label (e.g., event nodes)
    for uid, i in node_mapping.items():
        if uid in user_labels_df.index:
            y[i] = user_labels_df.loc[uid]['fraud_label']
    
    # --- 4. Assemble the SINGLE Data Object ---
    data = TemporalData(src=src, dst=dst, t=t, y=y)
    data.num_nodes = num_all_nodes
    # Track how many nodes correspond to users (as opposed to events),
    # and their original ids, so we can align external user features such
    # as SBERT review embeddings with the node indices.
    data.num_users = num_users
    data.user_ids = all_user_ids
    
    # --- 5. Data split: one canonical user-level split shared by every fraud method ---
    # This is the split source of truth (see notebooks/fraud_splits.py). Masks span users+events
    # but only user nodes are assigned; event nodes stay False (excluded downstream via y != -1),
    # so this reproduces the previously-cached temporal split exactly.
    from fraud_splits import get_canonical_user_split, masks_for_uids
    dataset_name = os.path.basename(base_path).replace('amazon-', '')
    seed = args.seed if args is not None else 0
    uid_split = get_canonical_user_split(base_path, dataset_name, seed)
    data.train_mask, data.val_mask, data.test_mask = masks_for_uids(uid_split, node_mapping, data.num_nodes)
    logging.info(
        "Canonical split: train=%d val=%d test=%d",
        int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum()),
    )

    return data


def init_user_embeddings_from_reviews(data, embedding_module, data_dir, args):
    """
    Initialize user node embeddings with SBERT-based review representations
    (as produced in review_embedding.py), for a fair comparison with the
    static-review models.
    """
    # We need to know how many nodes correspond to users and their original ids.
    num_users = getattr(data, "num_users", None)
    user_ids = getattr(data, "user_ids", None)
    if num_users is None or user_ids is None:
        logging.warning(
            "Data object does not have `num_users`/`user_ids` attributes; "
            "skipping SBERT-based initialization."
        )
        return

    try:
        # Reuse the cached embeddings pipeline from review_embedding.py.
        nodes_df = pd.read_csv(os.path.join(data_dir, "all_nodes.csv"))

        # Use the same SBERT model as in review_embedding.py.
        sbert_model_path = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        sbert_model = SentenceTransformer(sbert_model_path)

        # This will either load cached arrays or build them once.
        emb_array, mask_array, _, uid_to_idx = process_and_embed_user_events(
            nodes_df, sbert_model, args.max_events, data_dir
        )
    except Exception as e:
        logging.warning(
            "Could not load or build SBERT review embeddings (%s); "
            "falling back to random initialization.",
            e,
        )
        return

    # emb_array: (N_users_with_reviews, max_len, D)
    # mask_array: (N_users_with_reviews, max_len)
    # We pool over time to get a single D-dim vector per user.
    N_all, _, D = emb_array.shape
    logging.info(
        "Loaded SBERT review embeddings with shape (N=%d, T=?, D=%d). "
        "Building per-user pooled representations for %d users.",
        N_all,
        D,
        num_users,
    )

    user_text_emb = np.zeros((num_users, D), dtype=np.float32)

    for local_idx, uid in enumerate(user_ids):
        idx = uid_to_idx.get(uid, None)
        if idx is None:
            # User has no review-based embedding; leave as zeros.
            continue

        seq_emb = emb_array[idx]        # (T, D)
        seq_mask = mask_array[idx]      # (T,)
        if seq_mask.any():
            user_text_emb[local_idx] = seq_emb[seq_mask].mean(axis=0)

    device = embedding_module.weight.device
    user_text_emb_t = torch.from_numpy(user_text_emb).to(
        device=device, dtype=embedding_module.weight.dtype
    )

    # If SBERT dim does not match the configured embedding_dim, apply a small
    # linear projection once to map into the model's embedding space, then use
    # the result to initialize the embedding table.
    embedding_dim = embedding_module.embedding_dim
    if D != embedding_dim:
        logging.info(
            "Projecting SBERT dimension D=%d to embedding_dim=%d for initialization.",
            D,
            embedding_dim,
        )
        proj = torch.nn.Linear(D, embedding_dim, bias=False).to(device)
        torch.nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            user_init = proj(user_text_emb_t)
    else:
        user_init = user_text_emb_t

    with torch.no_grad():
        embedding_module.weight[:num_users].copy_(user_init)
    logging.info(
        "Initialized the first %d rows of embedding_module.weight with "
        "SBERT-based user review embeddings.",
        num_users,
    )


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
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
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
def evaluate(preds_proba, targets):
    """
    Computes a comprehensive set of classification metrics from prediction
    probabilities and ground-truth labels.
    
    Args:
        preds_proba (torch.Tensor): 1D tensor of positive class probabilities.
        targets (torch.Tensor): 1D tensor of ground-truth labels (0s and 1s).
    
    Returns:
        dict: A dictionary containing classification metrics.
    """
    # Convert tensors to NumPy arrays for sklearn and custom functions
    labels_np = targets.cpu().numpy().astype(int)
    probs_np = preds_proba.cpu().numpy()

    # --- Calculate Metrics (mirroring the test function) ---

    # 1. Accuracy
    # Get class predictions by using argmax on logits (equivalent to 0.5 threshold for 2 classes)
    preds_class_np = (probs_np >= 0.5).astype(int)
    acc = (preds_class_np == labels_np).mean()

    # 2. AUROC
    roc_auc = roc_auc_score(labels_np, probs_np)

    # 3. AUPRC
    precision, recall, _ = precision_recall_curve(labels_np, probs_np)
    pr_auc = auc(recall, precision)

    # 4. Custom Metrics from your original functions
    max_f1 = max_f1_score(labels_np, probs_np)
    recall_at_10_percent = recall_fraction(labels_np, probs_np, top_frac=0.1)
    precision_at_10_percent = top_precision(labels_np, probs_np, top_frac=0.1)

    # Assemble the final dictionary
    metrics = {
        'acc': acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'recall_at_10_percent': recall_at_10_percent,
        'precision_at_10_percent': precision_at_10_percent,
        'max_f1': max_f1,
    }
    
    return metrics

@torch.no_grad()
def compute_gnn_embeddings(model, data, device, embed_batch_size=4096):
    model['memory'].eval()
    model['gnn'].eval()
    model['embedding_module'].eval()
    
    all_node_ids = torch.arange(data.num_nodes, device=device)
    memory_embeddings, last_update = model['memory'](all_node_ids)
    
    num_nodes = data.num_nodes
    final_embeddings_list = []
    
    neighbor_loader = model['neighbor_loader']
    embedding_module = model['embedding_module']
    
    # REMOVED: neighbor_loader.reset_parameters()

    for start in range(0, num_nodes, embed_batch_size):
        end = min(start + embed_batch_size, num_nodes)
        batch_nodes = torch.arange(start, end, device=device)
        
        n_id, edge_index, e_id = neighbor_loader(batch_nodes)
        
        batch_src = data.src[e_id].to(device)
        batch_dst = data.dst[e_id].to(device)
        
        src_emb = embedding_module(batch_src)
        dst_emb = embedding_module(batch_dst)
        batch_msg = torch.cat([src_emb, dst_emb], dim=1)
        
        z = model['gnn'](
            x=memory_embeddings[n_id], 
            last_update=last_update[n_id], 
            edge_index=edge_index, 
            t=data.t[e_id].to(device), 
            msg=batch_msg
        )
        final_embeddings_list.append(z)

    return torch.cat(final_embeddings_list, dim=0)
@torch.no_grad()
def test(model, data, device, criterion, split="val", embed_batch_size=4096):
    model['nodeclf'].eval()

    # --- 1. Compute GNN Embeddings ---
    # This replaces the simple memory lookup with the full GNN pass
    final_node_embeddings = compute_gnn_embeddings(model, data, device, embed_batch_size)

    # --- 2. Prediction & Metrics ---
    # Select only the nodes relevant to this split (val or test)
    mask = data[f'{split}_mask'] & (data.y != -1)
    
    # Feed GNN embeddings into the classifier
    predictions_logits = model['nodeclf'](final_node_embeddings.to(device))
    loss = criterion(predictions_logits[mask], data.y[mask])
    
    labels = data.y[mask]
    preds_proba = F.softmax(predictions_logits[mask], dim=1)[:, 1]
    
    metrics = evaluate(preds_proba.cpu(), labels.cpu())
    metrics['loss'] = loss.item()
    
    return metrics
def train(model, data, optimizer, criterion,
          device, logger = None, args = None):
    """
    Train the model with batching and (optionally) early stopping.
    
    Args:
        model (dict): Dictionary containing the model modules.
        data (dict): Dictionary containing TemporalData objects.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        epochs (int): Number of training epochs.
        train_batch_size (int): Mini-batch size for training.
        eval_batch_size (int): Mini-batch size for evaluation.
        device (torch.device): Device for computations.
        patience (int): Early stopping patience (in epochs).
        eval_every (int): Evaluation frequency (in epochs).
        enable_early_stopping (bool): Whether to use early stopping.
        train_neg_sampling_ratio (float): Negative sampling ratio for training.
        hits_k (list): List of k values for computing Hits@k.
        
    Returns:
        tuple: (Updated model dictionary after training, best_test_metrics)
    """


    embedding_module = model['embedding_module']
    memory = model['memory']
    gnn = model['gnn']
    nodeclf = model['nodeclf']
    neighbor_loader = model['neighbor_loader']
    assoc = model['assoc']
    assoc2 = model['assoc2']
    
    embedding_module.train()
    memory.train()
    gnn.train()
    nodeclf.train()
    
    best_val_pr_auc = 0
    patience_counter = 0
    best_state = None
    best_test_metrics = None
    best_state = None

    logging.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # ====== 1. Memory Computation Stage ======
        # This block ensures every message is correctly formatted.
        memory.reset_state()
        
        # Create a clean data object for the loader to prevent other errors
        loader_data = TemporalData(src=data.src, dst=data.dst, t=data.t)
        loader = TemporalDataLoader(loader_data, batch_size=args.train_batch_size)
        
        for batch in tqdm(loader, desc=f"Epoch {epoch}: Computing memory", leave=False):
            batch = batch.to(device)
            with torch.no_grad():
                # Get 128-dim embeddings for source and destination nodes
                src_emb = embedding_module(batch.src)
                dst_emb = embedding_module(batch.dst)
                
                # Concatenate them to create a single 256-dim message tensor
                msg = torch.cat([src_emb, dst_emb], dim=1)
                

                memory.update_state(batch.src, batch.dst, batch.t, msg)
        # ====== 2. Node Classification & Training Stage ======
# ====== 2. Node Classification & Training Stage ======
        
        # Switch modules to training mode
        model['gnn'].train()
        model['nodeclf'].train()
        
        # REMOVED: model['neighbor_loader'].reset_parameters() <- This caused the crash
        
        # 1. Identify Training Nodes (We only compute loss on these)
        train_mask_idx = data.train_mask & (data.y != -1)
        train_nodes = torch.nonzero(train_mask_idx, as_tuple=True)[0]
        
        # 2. Sample Neighbors just for the training nodes
        #    n_id: all nodes involved (target + neighbors)
        #    edge_index: local graph structure
        #    e_id: edge IDs used for the GNN
        n_id, edge_index, e_id = model['neighbor_loader'](train_nodes)
        
        # 3. Get Memory for the sampled batch
        memory_embeddings, last_update = memory(n_id)
        
        # 4. Reconstruct Message (Msg) for the GNN edges
        src_emb_gnn = embedding_module(data.src[e_id].to(device))
        dst_emb_gnn = embedding_module(data.dst[e_id].to(device))
        gnn_msg = torch.cat([src_emb_gnn, dst_emb_gnn], dim=1)
        
        # 5. Run GNN
        #    Input: Memory (256 dim) -> Output: Z (128 dim)
        z = gnn(
            x=memory_embeddings, 
            last_update=last_update, 
            edge_index=edge_index, 
            t=data.t[e_id].to(device), 
            msg=gnn_msg
        )
        
        # 6. Extract Target Nodes
        #    The neighbor loader places the 'seed' (train) nodes first in the output
        z_train = z[:train_nodes.size(0)]
        
        # 7. Prediction & Loss
        predictions = nodeclf(z_train)
        loss = criterion(predictions, data.y[train_nodes])
        
        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Detach memory for the next epoch
        memory.detach()
        # avg_loss = total_loss / train_data.num_events
        avg_loss = loss.item()
        logging.info(f"Epoch {epoch:03d} Loss: {avg_loss:.4f}")
        
        if epoch % args.eval_every == 0:           
            val_metrics = test(model, data, device, criterion, split="val")
            
            # Always track best test metrics when validation improves
            if val_metrics['pr_auc'] > best_val_pr_auc:
                best_val_pr_auc = val_metrics['pr_auc']
                # Get test metrics when validation performance peaks
                test_metrics = test(model, data, device, criterion, split="test")
                best_test_metrics = test_metrics.copy()
                logging.info(f"New best validation PR-AUC: {best_val_pr_auc:.4f}")
                logging.info(f"Corresponding test metrics: {test_metrics}")
                
                # Save best model state
                best_state = {k: v.state_dict() for k, v in model.items() if hasattr(v, "state_dict")}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if logger:
                log_dict = {'epoch': epoch, 'train_loss': avg_loss,}
                for key, value in val_metrics.items():
                    log_dict[f'val_{key}'] = value
                logger.log(log_dict)
            
            if args.enable_early_stopping:
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break              

    if best_state is not None:
        for k, v in model.items():
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(best_state[k])
    
    return model, best_test_metrics


###############################################################################
# Checkpoint Saving Function
###############################################################################
def save_checkpoint(filepath, model, optimizer, args, elapsed_time, test_metrics, best_test_metrics=None):
    """
    Save the model, optimizer, training arguments, elapsed time,
    test metrics, and best test metrics to a checkpoint file.
    """
    checkpoint = {
        'experiment_name': args.experiment_name,
        'model_state_dict': {k: v.state_dict() for k, v in model.items() if hasattr(v, "state_dict")},
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'elapsed_time': elapsed_time,
        'test_metrics': test_metrics,
        'best_test_metrics': best_test_metrics,
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved at {filepath}")

###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    args = parse_args()

    
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    run_name = (
        f"{args.model}_nngb{args.num_neighbors}_hidden{args.embedding_dim}_"
        f"lr{args.lr}_bs{args.train_batch_size}_epochs{args.epochs}_seed{args.seed}"
    )
    
    project_name = f'temporal-graph-fraud-{args.dataset}'
    if 'GAT' in args.model:
        run_name += f"_heads{args.heads}"
    if args.use_sequence_embedding and not run_name.endswith("_sbert"):
        run_name += "_sbert"
    logger = LocalLogger(project_name=project_name, run_name=run_name, log_dir=args.log_dir)
    if logger.skip:
        print("Exiting script because the experiment is being skipped.")
        exit()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(script_dir, '..', 'data', 'processed', f'amazon-{args.dataset}')
    data_dir = os.path.join(args.data_root, f'amazon-{args.dataset}')
    data = load_data(data_dir, val_size=0.2, test_size=0.2, args=args)
    data = data.to(device) 
    num_all_nodes = data.num_nodes
    
    # model
    embedding_dim = args.embedding_dim
    msg_dim = memory_dim = time_dim = args.embedding_dim * 2
    
    embedding_module = torch.nn.Embedding(num_all_nodes, embedding_dim).to(device)
    # Optionally initialize user node embeddings from SBERT review representations.
    if args.use_sequence_embedding:
        init_user_embeddings_from_reviews(data, embedding_module, data_dir, args)

    if args.model == 'TGN':
        memory = TGNMemory(
            num_all_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)
    elif args.model == 'DyRep':
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

    # linkpred = LinkPredictor(in_channels=embedding_dim).to(device)
    # Node classifier operates on memory embeddings whose size is `memory_dim`.
    nodeclf = MLPClassifier(embedding_dim, memory_dim, 2, num_layers=args.clf_layers).to(device)
    neighbor_loader = LastNeighborLoader(num_all_nodes, size=args.num_neighbors, device=device)
    
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(num_all_nodes, dtype=torch.long, device=device)
    assoc2 = torch.empty(num_all_nodes, dtype=torch.long, device=device)
    
    model = {
        'embedding_module': embedding_module,
        'memory': memory,
        'gnn': gnn,
        # 'linkpred': linkpred,
        'nodeclf': nodeclf,
        'neighbor_loader': neighbor_loader,
        'assoc': assoc,
        'assoc2': assoc2
    }
    
    optimizer = torch.optim.Adam(
        set(memory.parameters()) |
        set(gnn.parameters()) |
        set(nodeclf.parameters()) |
        set(embedding_module.parameters()),
        lr=args.lr
    )
    criterion = F.cross_entropy

    start_time = time.time()
    model, best_test_metrics = train(
        model, data, optimizer, criterion, device, logger = logger,
        args = args
    )
    elapsed_time = time.time() - start_time
    # val_metrics = test(model, data, device, criterion, split="val")
    val_metrics = test(model, data, device, criterion, split="val", embed_batch_size=args.eval_batch_size)
    test_metrics = test(model, data, device, criterion, split="test")
    
    # Log best test metrics if available
    if best_test_metrics is not None:
        logging.info("--- Best Test Results (when validation peaked) ---")
        best_test_log_msg = "Best Test Metrics | "
        best_test_log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in best_test_metrics.items()])
        logging.info(best_test_log_msg)
    
    # --- ADD THIS BLOCK FOR FINAL LOGGING ---
    logging.info("--- Final Evaluation Results ---")

    # 1. Log Final Validation Metrics to Console
    val_log_msg = "Final Validation Metrics | "
    val_log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
    logging.info(val_log_msg)

    # 2. Log Final Test Metrics to Console
    test_log_msg = "Final Test Metrics     | "
    test_log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
    logging.info(test_log_msg)

    # 3. Prepare a dictionary for structured logging (to file and wandb)
    final_results_dict = {}
    for key, value in val_metrics.items():
        final_results_dict[f'val_{key}'] = value
    for key, value in test_metrics.items():
        final_results_dict[f'test_{key}'] = value
    final_results_dict['train_wall_sec'] = elapsed_time

    # 4. Log to local file
    if logger:
        logger.log(final_results_dict)
        logger.finish()


    # --- END OF LOGGING BLOCK ---
    

    os.makedirs("stored", exist_ok=True)
    experiment_name = args.experiment_name
    if args.use_sequence_embedding and not experiment_name.endswith("_sbert"):
        experiment_name += "_sbert"
    checkpoint_filename = f"stored/{experiment_name}_checkpoint.pt"
    save_checkpoint(checkpoint_filename, model, optimizer, args, elapsed_time, test_metrics, best_test_metrics)