import os
import time
import random
import argparse
import logging
from tqdm import tqdm

import pickle
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, TransformerConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# import wandb
os.environ.setdefault("WANDB_MODE", "disabled")  # box public W&B is 403-blocked; never upload
import wandb
from review_embedding import *
import json
from review_embedding import LocalLogger
from sentence_transformers import SentenceTransformer


def save_data_splits(data, split_dir, dataset_name, seed):
    """
    Save train/val/test splits to disk for consistent reuse.
    """
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, f'{dataset_name}_splits_seed{seed}.pt')
    
    splits = {
        'train_mask': data.train_mask.cpu(),
        'val_mask': data.val_mask.cpu(), 
        'test_mask': data.test_mask.cpu()
    }
    
    torch.save(splits, split_file)
    logging.info(f"Saved data splits to: {split_file}")
    return split_file


def load_data_splits(split_dir, dataset_name, seed, device):
    """
    Load pre-saved train/val/test splits from disk.
    Returns the splits if found, None otherwise.
    """
    split_file = os.path.join(split_dir, f'{dataset_name}_splits_seed{seed}.pt')
    
    if os.path.exists(split_file):
        logging.info(f"Loading pre-saved data splits from: {split_file}")
        splits = torch.load(split_file, map_location=device)
        return splits['train_mask'], splits['val_mask'], splits['test_mask']
    else:
        logging.info(f"No pre-saved splits found at: {split_file}")
        return None, None, None

###############################################################################
# Logging configuration
###############################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Node Classification with Batching on Large Graphs"
    )
    parser.add_argument("--dataset", type=str, choices=['musical', 'movies', 'clothing', 'electronics', 'home-kitchen', 'office-products'], default='musical')
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"),
                        help="dir with amazon-<dataset>/ (all_nodes.csv, all_edges.csv, splits/)")
    parser.add_argument("--log_dir", type=str, default="logs_delta",
                        help="root dir for LocalLogger JSONs; use a fresh dir for the canonical rerun")
    parser.add_argument('--model', type=str, default='GCN',
                        help="Model type to use.")
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help="Number of hidden channels in each layer.")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="Number of GNN layers.")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help="Dimensionality of input node embeddings if no features are provided.")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Maximum number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate.")
    parser.add_argument('--heads', type=int, default=2,
                        help="Number of attention heads for GAT.")
    parser.add_argument('--train_batch_size', type=int, default=2**12,
                        help="Mini-batch size for training.")
    parser.add_argument('--eval_batch_size', type=int, default=2**12,
                        help="Mini-batch size for evaluation and testing.")
    parser.add_argument('--patience', type=int, default=20,
                        help="Early stopping patience (in epochs).")
    parser.add_argument('--eval_every', type=int, default=5,
                        help="Evaluation frequency (in epochs).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU ID to use (if CUDA is available).")
    parser.add_argument('--enable_early_stopping', action='store_true',
                        help='Enable early stopping based on validation accuracy.')
    parser.add_argument('--num_neighbors', type=int, default=10,
                        help="Number of neighbors to sample per layer in NeighborLoader.")
    parser.add_argument('--experiment_name', type=str, default='exp-1',
                        help="Experiment name for saving checkpoints and artifacts.")
    parser.add_argument('--use_sequence_embedding', action='store_true',
                        help='Use pre-computed sequence embedding as node features.')
    parser.add_argument('--seq_embedding_filename', type=str, default='user_embedding.pkl',
                        help="Filename storing the sequence embedding.")
    # --- ADDED: Argument for using wandb ---
    parser.add_argument('--use_wandb', action='store_true',
                        help="Enable logging to Weights & Biases.")
    parser.add_argument(
        '--suffix', 
        type=str, 
        default='', 
        help="suffix to wandb project"
    )
    parser.add_argument('--force_new_splits', action='store_true', default=False, 
                        help="Force creation of new data splits even if existing ones are found")
    parser.add_argument('--val_size', type=float, default=0.1,
                        help="Fraction of nodes to use for validation")
    parser.add_argument('--test_size', type=float, default=0.1,
                        help="Fraction of nodes to use for testing")
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
def load_data(base_path, embedding_dim, use_sequence_embedding=False, seq_embedding_filename=None, 
              dataset_name=None, seed=42, force_new_splits=False, val_size=0.1, test_size=0.1, device=None):
    """
    Load node and edge data from CSV files and create a PyG Data object.
    """
    logging.info("Preparing dataset...")
    
    nodes_path = os.path.join(base_path, "all_nodes.csv")
    edges_path = os.path.join(base_path, "all_edges.csv")
    
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    node_mapping = {node_id: i for i, node_id in enumerate(nodes_df['uid'].unique())}
    
    source_nodes = edges_df['uid'].map(node_mapping)
    target_nodes = edges_df['other_uid'].map(node_mapping)
    
    # Drop rows where mapping resulted in NaN
    source_nodes.dropna(inplace=True)
    target_nodes.dropna(inplace=True)
    
    # Align indices
    common_indices = source_nodes.index.intersection(target_nodes.index)
    source_nodes = source_nodes[common_indices]
    target_nodes = target_nodes[common_indices]
    
    edge_index = torch.tensor([source_nodes.values, target_nodes.values], dtype=torch.long)
    
    num_nodes = len(node_mapping)
    
    if use_sequence_embedding:
        # Use Sentence-BERT to build per-user embeddings as in review_embedding.py
        logging.info("Using Sentence-BERT to compute node embeddings (use_sequence_embedding=True).")

        # SBERT model is loaded lazily inside process_and_embed_user_events — only on a cache MISS.
        # With a warm embedding cache (all_embeddings.npy), the model path is never touched.
        sbert_model_path = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # Reuse the streaming + cached embedding pipeline from review_embedding.py
        # This returns per-event embeddings of shape (N_users, max_len, D) and a uid_to_idx map.
        max_events = 50  # keep consistent with review_embedding.py default
        embeddings_array, masks_array, _, uid_to_idx = process_and_embed_user_events(
            nodes_df, lambda: SentenceTransformer(sbert_model_path), max_events, base_path
        )

        # Convert sequence embeddings into a single static embedding per user
        # via masked mean pooling over the time dimension.
        masks = masks_array.astype(bool)
        valid_counts = np.clip(masks.sum(axis=1, keepdims=True), a_min=1, a_max=None)
        pooled = (embeddings_array * masks[..., None]).sum(axis=1) / valid_counts

        x = torch.zeros((num_nodes, pooled.shape[1]), dtype=torch.float32)
        for original_id, new_id in node_mapping.items():
            if original_id in uid_to_idx:
                idx = uid_to_idx[original_id]
                x[new_id] = torch.from_numpy(pooled[idx]).float()
    else:
        x = None
 
    # all_nodes.csv is per-EVENT (many rows per uid), so `.values` would index labels by ROW, not
    # by user node id -> smeared/misaligned labels. Take one label per uid and align to node_mapping.
    uid_label = nodes_df.groupby('uid', sort=False)['fraud_label'].first().to_dict()
    labels = torch.tensor([int(uid_label[uid]) for uid in node_mapping], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes)
    
    # One canonical user-level split shared by every fraud method (see notebooks/fraud_splits.py).
    from fraud_splits import get_canonical_user_split, masks_for_uids
    uid_split = get_canonical_user_split(base_path, dataset_name, seed)
    train_mask, val_mask, test_mask = masks_for_uids(uid_split, node_mapping, num_nodes)
    data.train_mask = train_mask.to(device) if device else train_mask
    data.val_mask = val_mask.to(device) if device else val_mask
    data.test_mask = test_mask.to(device) if device else test_mask
    
    data = T.ToUndirected()(data)
    logging.info(f"Data object created: {data}")
    logging.info(f"Train nodes: {data.train_mask.sum()}, Val nodes: {data.val_mask.sum()}, Test nodes: {data.test_mask.sum()}")
    
    return data

###############################################################################
# Model Definitions
###############################################################################
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.layers.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.layers.append(GATv2Conv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(GINConv(nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))))
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))))
        self.layers.append(GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class TFConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(TransformerConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.layers.append(TransformerConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class Model(torch.nn.Module):
    """Wrapper model for node classification."""
    def __init__(self, gnn, embedding_dim, num_nodes, use_sequence_embedding):
        super().__init__()
        self.gnn = gnn
        self.use_sequence_embedding = use_sequence_embedding
        if not use_sequence_embedding:
            self.emb = torch.nn.Embedding(num_nodes, embedding_dim)

    def forward(self, x, edge_index):
        if not self.use_sequence_embedding and x is not None:
            x = self.emb(x)
        out = self.gnn(x, edge_index)
        return out

###############################################################################
# Evaluation Function for Node Classification
###############################################################################
@torch.no_grad()
def test(model, loader, device):
    """Evaluate model performance on a given split for node classification."""
    model.eval()
    
    all_preds, all_labels, all_probas = [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        input_features = batch.x if model.use_sequence_embedding else batch.n_id
        out = model(input_features, batch.edge_index)
        
        batch_size = batch.batch_size
        out_seeds, labels = out[:batch_size], batch.y[:batch_size]
        
        preds = out_seeds.argmax(dim=-1)
        probas = F.softmax(out_seeds, dim=-1)[:, 1]

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probas.append(probas.cpu())

    final_preds, final_labels, final_probas = torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(), torch.cat(all_probas).numpy()
    
    # --- MODIFIED: Calculate all specified metrics ---
    acc = accuracy_score(final_labels, final_preds)
    f1 = f1_score(final_labels, final_preds, average='weighted', zero_division=0)
    try:
        auc_score = roc_auc_score(final_labels, final_probas)
    except ValueError:
        auc_score = 0.5

    precision, recall, _ = precision_recall_curve(final_labels, final_probas)
    pr_auc = auc(recall, precision)
    recall_at_10_percent = recall_fraction(final_labels, final_probas, top_frac=0.1)
    precision_at_10_percent = top_precision(final_labels, final_probas, top_frac=0.1)
    max_f1 = max_f1_score(final_labels, final_probas)

    metrics = {
        'acc': acc,
        'roc_auc': auc_score,
        'pr_auc': pr_auc,
        'recall_at_10_percent': recall_at_10_percent,
        'precision_at_10_percent': precision_at_10_percent,
        'max_f1': max_f1,
    }
    
    return metrics
###############################################################################
# Training Loop for Node Classification
###############################################################################
def train(model, data, optimizer, epochs, train_batch_size, eval_batch_size,
          device, patience, eval_every, enable_early_stopping, num_neighbors,logger = None):
    """Train the model for node classification."""
    
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                  num_neighbors=[num_neighbors] * model.gnn.num_layers,
                                  batch_size=train_batch_size, shuffle=True)
    val_loader = NeighborLoader(data, input_nodes=data.val_mask,
                                num_neighbors=[num_neighbors] * model.gnn.num_layers,
                                batch_size=eval_batch_size, shuffle=False)

    # Class-weighted loss to match review_embedding.py (seq-only / BRIDGE): on the heavily
    # imbalanced fraud task, unweighted CE lets the majority class swamp the gradient and the
    # scores collapse toward the base rate. weight = [1.0, num_neg/num_pos].
    # data.y stays on CPU (NeighborLoader moves batches per-iter); mask may be on device -> index on CPU.
    train_labels = data.y[data.train_mask.cpu()]
    num_class_0 = (train_labels == 0).sum().item()
    num_class_1 = (train_labels == 1).sum().item()
    class_weights = torch.tensor([1.0, num_class_0 / max(num_class_1, 1)],
                                 dtype=torch.float, device=device)
    logging.info(f"Class weights for loss: {class_weights.tolist()}")
    
    best_val_pr_auc = -1
    patience_counter = 0
    best_state = None
    best_test_metrics = None

    logging.info("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_acc = 0, 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            input_features = batch.x if model.use_sequence_embedding else batch.n_id
            out = model(input_features, batch.edge_index)
            
            batch_size = batch.batch_size
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size], weight=class_weights)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred = out[:batch_size].argmax(dim=1)
                acc = (pred == batch.y[:batch_size]).sum().float() / batch_size
                total_acc += acc.item()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        
        if epoch % eval_every == 0:
            val_metrics = test(model, val_loader, device)
            logging.info(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f} | Val Acc: {val_metrics['acc']:.4f}, Val F1: {val_metrics['max_f1']:.4f}, Val PR AUC: {val_metrics['pr_auc']:.4f}")
            # --- MODIFIED: Use the generic logger ---
            if logger:
                log_dict = {'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_acc}
                for key, value in val_metrics.items():
                    log_dict[f'val_{key}'] = value
                logger.log(log_dict)


            # Track best test metrics by val PR-AUC (matches BRIDGE/temporal/RelGT selection).
            if val_metrics['pr_auc'] > best_val_pr_auc:
                best_val_pr_auc = val_metrics['pr_auc']
                
                # Save test metrics when validation is best
                test_loader_for_best = NeighborLoader(data, input_nodes=data.test_mask,
                                                     num_neighbors=[num_neighbors] * model.gnn.num_layers,
                                                     batch_size=eval_batch_size, shuffle=False)
                test_metrics_current = test(model, test_loader_for_best, device)
                best_test_metrics = test_metrics_current.copy()
                
                # Handle early stopping logic if enabled
                if enable_early_stopping:
                    best_state = model.state_dict()
                    patience_counter = 0
            elif enable_early_stopping:
                patience_counter += 1
                
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_test_metrics

###############################################################################
# Checkpoint Saving Function
###############################################################################
def save_checkpoint(filepath, model, optimizer, args, elapsed_time, test_metrics):
    """Save model checkpoint."""
    checkpoint = {
        'experiment_name': args.experiment_name,
        'model_state_dict': model.state_dict(),
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
if __name__ == "__main__":
    args = parse_args()

    # --- ADDED: W&B Initialization ---
    run_name = (
        f"{args.model}_layers{args.num_layers}_hidden{args.hidden_channels}_"
        f"lr{args.lr}_bs{args.train_batch_size}_epochs{args.epochs}_seed{args.seed}"
    )
    if 'GAT' in args.model or 'Transformer' in args.model:
        # run_name += f"_heads{args.heads}"
        run_name=run_name.replace(f"_seed{args.seed}", f"_heads{args.heads}_seed{args.seed}")
    if args.use_sequence_embedding:
        run_name += "_sbert"
    
    project_name = f'static-graph-fraud-{args.dataset}'
    logger = LocalLogger(project_name=project_name, run_name=run_name, log_dir=args.log_dir)
    if logger.skip:
        # print("Exiting script because the experiment is being skipped.")
        exit() # This will stop the main function from continuing
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(args.data_root, f'amazon-{args.dataset}')
    
    data = load_data(data_dir, args.embedding_dim, args.use_sequence_embedding, args.seq_embedding_filename,
                     dataset_name=args.dataset, seed=args.seed, force_new_splits=args.force_new_splits,
                     val_size=args.val_size, test_size=args.test_size, device=device)
    
    num_classes = len(data.y.unique())
    in_channels = data.num_node_features if args.use_sequence_embedding else args.embedding_dim

    model_classes = {'GCN': GCN, 'GraphSAGE': GraphSAGE, 'GAT': GAT, 'GIN': GIN, 'GATv2': GATv2, 'TransformerConv': TFConv}
    
    gnn_args = {
        'in_channels': in_channels,
        'hidden_channels': args.hidden_channels,
        'out_channels': num_classes,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    if 'GAT' in args.model or 'Transformer' in args.model:
        gnn_args['heads'] = args.heads

    gnn = model_classes[args.model](**gnn_args)
        
    model = Model(gnn, args.embedding_dim, data.num_nodes, args.use_sequence_embedding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    
    model, best_test_metrics = train(
        model, data, optimizer, args.epochs,
        args.train_batch_size, args.eval_batch_size, device,
        patience=args.patience, eval_every=args.eval_every,
        enable_early_stopping=args.enable_early_stopping,
        num_neighbors=args.num_neighbors,
        logger = logger
    )
    elapsed_time = time.time() - start_time
    
    val_loader = NeighborLoader(data, input_nodes=data.val_mask,
                            num_neighbors=[args.num_neighbors] * model.gnn.num_layers,
                            batch_size=args.eval_batch_size, shuffle=False)
    val_metrics = test(model, val_loader, device)

    if args.use_wandb:
        wandb.log({"val_accuracy": val_metrics['acc'], "val_max_f1": val_metrics['max_f1'],"val_roc_auc": val_metrics['roc_auc'], "val_pr_auc": val_metrics['pr_auc']})
        
    if logger:
        log_dict = {'epoch': args.epochs}
        for key, value in val_metrics.items():
            log_dict[f'val_{key}'] = value
        logger.log(log_dict)
        
    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
    
    
    test_metrics = test(model, test_loader, device)
    logging.info(f"Final Test Metrics | Test Acc: {test_metrics['acc']:.4f}, Test F1: {test_metrics['max_f1']:.4f}, Test PR AUC: {test_metrics['pr_auc']:.4f}")

    if args.use_wandb:
        wandb_log = {"test_accuracy": test_metrics['acc'], "test_max_f1": test_metrics['max_f1'],"test_roc_auc": test_metrics['roc_auc'], "test_pr_auc": test_metrics['pr_auc']}
        # Also log the best test metrics (when validation was best)
        if best_test_metrics:
            for key, value in best_test_metrics.items():
                wandb_log[f'best_test_{key}'] = value
        wandb.log(wandb_log)
        wandb.finish()
        
    if logger:
        log_dict = {'epoch': args.epochs}
        for key, value in test_metrics.items():
            log_dict[f'test_{key}'] = value
        # Also log the best test metrics (when validation was best)
        # breakpoint()
        if best_test_metrics:
            for key, value in best_test_metrics.items():
                log_dict[f'best_test_{key}'] = value
        logger.log(log_dict)
        
    print("\nClassification Report (Validation Set):")
    # print(classification_report(data.y[data.val_mask].cpu(), val_pred))

    if logger:
        logger.log({"train_wall_sec": elapsed_time})
        logger.finish()

    os.makedirs("stored/", exist_ok=True)
    checkpoint_filename = f"stored/{args.experiment_name}_node_clf_checkpoint.pt"
    save_checkpoint(checkpoint_filename, model, optimizer, args, elapsed_time, test_metrics)