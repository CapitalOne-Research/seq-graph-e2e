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
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.loader import LinkNeighborLoader
from torcheval.metrics import ReciprocalRank, HitRate

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
    parser.add_argument('--base_path', type=str, default='.',
                        help="Base path for CSV files.")
    parser.add_argument('--model', type=str, choices=['GCN', 'GraphSAGE', 'GAT', 'GIN'], default='GCN',
                        help="Model type to use.")
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help="Number of hidden channels in each layer.")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="Number of GNN layers.")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help="Dimensionality of input node embeddings.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate.")
    parser.add_argument('--heads', type=int, default=4,
                        help="Number of attention heads for GAT.")
    parser.add_argument('--train_batch_size', type=int, default=2**12,
                        help="Mini-batch size for training.")
    parser.add_argument('--eval_batch_size', type=int, default=2**20,
                        help="Mini-batch size for evaluation and testing.")
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience (in epochs).")
    parser.add_argument('--eval_every', type=int, default=5,
                        help="Evaluation frequency (in epochs).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU ID to use (if CUDA is available).")
    # New flag: early stopping is disabled by default
    parser.add_argument('--enable_early_stopping', action='store_true',
                        help='Enable early stopping based on validation MRR. (default: False)')
    # New arguments for neighbor loader settings.
    parser.add_argument('--num_neighbors', type=int, default=10,
                        help="Number of neighbors to sample per layer in LinkNeighborLoader.")
    parser.add_argument('--train_neg_sampling_ratio', type=float, default=5,
                        help="Negative sampling ratio for training in LinkNeighborLoader.")
    # New argument for custom Hits@k values.
    parser.add_argument('--hits_k', type=str, default="3,5,10,30",
                        help="Comma separated list of k values for computing Hits@k metrics.")
    # New argument for experiment name (for saving artifacts)
    parser.add_argument('--experiment_name', type=str, default='exp-1',
                        help="Experiment name for saving checkpoints and artifacts.")
    # Sequence embedding flag
    parser.add_argument('--use_sequence_embedding', action='store_true',
                        help='Use sequence embedding as additional input to the graph')
    parser.add_argument('--seq_embedding_filename', type=str, default='user_embedding.pkl',
                        help="Filename storing the sequence embedding.")
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
def load_data(base_path, embedding_dim, use_sequence_embedding=False, seq_embedding_filename=None):
    """
    Load edge lists from CSV files with a 'relational_' prefix and create a PyG Data object.
    
    The files are expected to be:
      - relational_train.csv
      - relational_val.csv
      - relational_test.csv
      - relational_val_negative_sample.csv
      - relational_test_negative_sample.csv
      
    Args:
        base_path (str): Base directory where the CSV files are located.
        embedding_dim (int): Dimensionality of node embeddings.
        
    Returns:
        Data: A PyG Data object with custom attributes.
    """
    
    logging.info("Preparing dataset...")
    logging.info(f"base path: {base_path}")
    
    train_path = os.path.join(base_path, "relational_train.csv")
    val_path = os.path.join(base_path, "relational_val.csv")
    test_path = os.path.join(base_path, "relational_test.csv")
    val_neg_path = os.path.join(base_path, "relational_val_negative_sample.csv")
    test_neg_path = os.path.join(base_path, "relational_test_negative_sample.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    val_neg_df = pd.read_csv(val_neg_path)
    test_neg_df = pd.read_csv(test_neg_path)

    train_edge_index = torch.tensor([train_df['uid'].values, train_df['other_uid'].values], dtype=torch.long)
    val_edge_index = torch.tensor([val_df['uid'].values, val_df['other_uid'].values], dtype=torch.long)
    test_edge_index = torch.tensor([test_df['uid'].values, test_df['other_uid'].values], dtype=torch.long)
    val_neg_edge_index = torch.tensor([val_neg_df['uid'].values, val_neg_df['other_uid'].values], dtype=torch.long)
    test_neg_edge_index = torch.tensor([test_neg_df['uid'].values, test_neg_df['other_uid'].values], dtype=torch.long)

    num_nodes = max(
        train_df['uid'].max(), train_df['other_uid'].max(),
        val_df['uid'].max(), val_df['other_uid'].max(),
        test_df['uid'].max(), test_df['other_uid'].max()
    ) + 1
    
    if use_sequence_embedding:
        with open(seq_embedding_filename, 'rb') as file:
            seq_emb_dict = pickle.load(file)
            
        x = torch.zeros((num_nodes, embedding_dim))
        for i in range(num_nodes):
            if i in seq_emb_dict:
                x[i] = torch.Tensor(seq_emb_dict[i]).float()
                
        data = Data(
            x=x,
            node_id=torch.arange(num_nodes),
            edge_index=train_edge_index,
            val_edge_index=val_edge_index,
            test_edge_index=test_edge_index,
            val_neg_edge_index=val_neg_edge_index,
            test_neg_edge_index=test_neg_edge_index
        )
    else:        
        data = Data(
            node_id=torch.arange(num_nodes),
            edge_index=train_edge_index,
            val_edge_index=val_edge_index,
            test_edge_index=test_edge_index,
            val_neg_edge_index=val_neg_edge_index,
            test_neg_edge_index=test_neg_edge_index
        )
    
    data = T.ToUndirected()(data)
    
    print(data)
    
    return data

###############################################################################
# Model Definitions
###############################################################################
class GCN(torch.nn.Module):
    """Graph Convolutional Network."""
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    """GraphSAGE model."""
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, edge_index)
        return x

class GAT(torch.nn.Module):
    """Graph Attention Network."""
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, heads):
        super().__init__()
        assert hidden_channels%heads == 0, "hidden_channels must be divisible by heads for GAT"
        self.layers = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=dropout)
        ])
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, edge_index)
        return x
    
class GIN(torch.nn.Module):
    """Graph Isomorphism Network."""
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            GINConv(torch.nn.Linear(in_channels, hidden_channels))
        ])
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(torch.nn.Linear(hidden_channels, hidden_channels)))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, edge_index)
        return x

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
    
class Model(torch.nn.Module):
    """Model"""
    def __init__(self, gnn, embedding_dim, hidden_channels, num_nodes, use_sequence_embedding):
        super().__init__()
        self.gnn = gnn
        self.linkpred = LinkPredictor(hidden_channels)
        self.emb = torch.nn.Embedding(num_nodes, embedding_dim)
        self.use_sequence_embedding = use_sequence_embedding

    def forward(self, batch):
        x = self.emb(batch.node_id)
        if self.use_sequence_embedding:
            x = torch.cat([x, batch.x], dim=1)
        out = self.gnn(x, batch.edge_index)
        return out

###############################################################################
# Evaluation Functions using torcheval
###############################################################################
def evaluate(candidates, target, k_list):
    """
    Compute retrieval metrics using torcheval.metrics.
    
    Args:
        candidates (torch.Tensor): Tensor of shape (num_samples, num_candidates)
            containing scores for each candidate edge (with the positive at index 0).
        target (torch.Tensor): 1D tensor of shape (num_samples,) where each element is the index 
            of the positive candidate (0 in our case).
        k_list (list): List of k values for computing Hits@k.
    
    Returns:
        dict: A dictionary with keys "MRR" and "Hits@{k}" for each k.
    """
    mrr_metric = ReciprocalRank()
    hits_metrics = {k: HitRate(k=k) for k in k_list}
    
    mrr_metric.update(candidates, target)
    for k, hits_metric in hits_metrics.items():
        hits_metric.update(candidates, target)
    
    mrr = mrr_metric.compute().mean().item()
    hits_results = {f"Hits@{k}": hits_metrics[k].compute().mean().item() for k in k_list}
    return {"MRR": mrr, **hits_results}

def test(model, predictor, data, eval_batch_size, device, num_neighbors, hits_k, split="test"):
    """
    Evaluate model performance on a given split using torcheval.metrics.
    
    Args:
        model (torch.nn.Module): The GNN model.
        predictor (torch.nn.Module): The link predictor module.
        data (Data): The PyG Data object.
        eval_batch_size (int): Mini-batch size for evaluation.
        device (torch.device): Device to run computations on.
        num_neighbors (int): Number of neighbors to sample per layer.
        hits_k (list): List of k values for computing Hits@k.
        split (str): Which split to evaluate on; "test" or "val".
    
    Returns:
        dict: A dictionary with evaluation metrics.
    """
    model.eval()
    pos_scores_list = []
    neg_scores_list = []
    
    if split == "val":
        pos_edge_index = data.val_edge_index
        neg_edge_index = data.val_neg_edge_index
        desc_prefix = "Validation"
    else:
        pos_edge_index = data.test_edge_index
        neg_edge_index = data.test_neg_edge_index
        desc_prefix = "Testing"
    
    pos_loader = LinkNeighborLoader(
        data,
        num_neighbors=[num_neighbors] * len(model.gnn.layers),
        edge_label_index=pos_edge_index,
        batch_size=eval_batch_size,
        shuffle=False,
        neg_sampling_ratio=0
    )
    neg_loader = LinkNeighborLoader(
        data,
        num_neighbors=[num_neighbors] * len(model.gnn.layers),
        edge_label_index=neg_edge_index,
        batch_size=eval_batch_size,
        shuffle=False,
        neg_sampling_ratio=0
    )
    
    with torch.no_grad():
        for batch in tqdm(pos_loader, desc=f"{desc_prefix} Pos", leave=False):
            batch.val_edge_index = None; batch.test_edge_index = None
            batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
            
            batch = batch.to(device)
            output = model(batch)
            scores = predictor(output[batch.edge_label_index[0]], output[batch.edge_label_index[1]]).squeeze()
            pos_scores_list.append(scores.cpu())
            
        for batch in tqdm(neg_loader, desc=f"{desc_prefix} Neg", leave=False):
            batch.val_edge_index = None; batch.test_edge_index = None
            batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
            
            batch = batch.to(device)
            output = model(batch)
            scores = predictor(output[batch.edge_label_index[0]], output[batch.edge_label_index[1]]).squeeze()
            neg_scores_list.append(scores.cpu())
    
    pos_scores = torch.cat(pos_scores_list, dim=0)
    neg_scores = torch.cat(neg_scores_list, dim=0)
    
    # note that the positive and negative samples are in the right order in the data frame
    # this pairing of pos and neg sample are valid
    num_samples = pos_scores.shape[0]
    num_neg = neg_scores.shape[0] // num_samples
    neg_scores = neg_scores.reshape(num_samples, num_neg)
    
    candidates = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    target = torch.zeros(num_samples, dtype=torch.long) # positive sample is always in 0 idx
    
    metrics = evaluate(candidates, target, k_list=hits_k)
    return metrics

###############################################################################
# Training with Batching and Early Stopping
###############################################################################
def train(model, predictor, data, optimizer, epochs, train_batch_size, eval_batch_size,
          device, patience, eval_every, enable_early_stopping, num_neighbors, train_neg_sampling_ratio, hits_k):
    """
    Train the model using mini-batches with early stopping based on validation MRR.
    If early stopping is disabled (default), the training will run for all epochs.
    Uses a LinkNeighborLoader with a customizable negative sampling ratio.
    Additionally, the average loss is printed after every epoch while evaluation metrics
    are logged only every 'eval_every' epochs.
    """
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[num_neighbors] * len(model.gnn.layers),
        edge_label_index=data.edge_index,
        batch_size=train_batch_size,
        shuffle=True,
        neg_sampling_ratio=train_neg_sampling_ratio
    )
    
    best_val_mrr = -float('inf')
    patience_counter = 0
    best_state = None

    logging.info("Starting training...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            batch.val_edge_index = None; batch.test_edge_index = None
            batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
            
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            scores = predictor(output[batch.edge_label_index[0]], output[batch.edge_label_index[1]]).squeeze()
            loss = F.binary_cross_entropy_with_logits(scores, batch.edge_label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch:03d} Loss: {avg_loss:.4f}")
        
        if epoch % eval_every == 0:
            metrics = test(model, predictor, data, eval_batch_size, device, num_neighbors, hits_k, split="val")
            val_mrr = metrics["MRR"]
            log_msg = f"Epoch {epoch:03d} Evaluation | Val MRR: {val_mrr:.4f}"
            for k in hits_k:
                log_msg += f", Hits@{k}: {metrics[f'Hits@{k}']:.4f}"
            logging.info(log_msg)
            
            if enable_early_stopping:
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    best_state = model.state_dict()
                    patience_counter = 0
                    logging.info("New best model found; resetting patience counter.")
                else:
                    patience_counter += 1
                    logging.info(f"No improvement; patience counter: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

###############################################################################
# Checkpoint Saving Function
###############################################################################
def save_checkpoint(filepath, model, predictor, optimizer, args, elapsed_time, test_metrics):
    """
    Save the model, predictor, optimizer, training arguments, elapsed time,
    and test metrics to a checkpoint file.
    """
    checkpoint = {
        'experiment_name': args.experiment_name,
        'model_state_dict': model.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
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
    # Process the custom hits_k argument as a list of integers.
    args.hits_k = [int(x) for x in args.hits_k.split(",")]
    
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Load data.
    data = load_data(args.base_path, args.embedding_dim, args.use_sequence_embedding, args.seq_embedding_filename)
    
    model_classes = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT,
        'GIN': GIN
    }
    
    if args.model == 'GAT':
        gnn = GAT(
            in_channels=(2 * args.embedding_dim if args.use_sequence_embedding else args.embedding_dim),
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            heads=args.heads
        )
    else:
        gnn = model_classes[args.model](
            in_channels=(2 * args.embedding_dim if args.use_sequence_embedding else args.embedding_dim),
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    model = Model(gnn, args.embedding_dim, args.hidden_channels, data.num_nodes, args.use_sequence_embedding).to(device)
    predictor = LinkPredictor(args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    model = train(
        model, predictor, data, optimizer, args.epochs,
        args.train_batch_size, args.eval_batch_size, device,
        patience=args.patience, eval_every=args.eval_every,
        enable_early_stopping=args.enable_early_stopping,
        num_neighbors=args.num_neighbors,
        train_neg_sampling_ratio=args.train_neg_sampling_ratio,
        hits_k=args.hits_k
    )
    elapsed_time = time.time() - start_time

    test_metrics = test(model, predictor, data, args.eval_batch_size, device, args.num_neighbors, args.hits_k, split="test")
    log_msg = f"{args.model}: MRR: {test_metrics['MRR']:.4f}"
    for k in args.hits_k:
        log_msg += f", Hits@{k}: {test_metrics[f'Hits@{k}']:.4f}"
    log_msg += f", Time: {elapsed_time:.2f}s"
    logging.info(log_msg)

    # Save the checkpoint using the experiment name.
    checkpoint_filename = f"stored/{args.experiment_name}_checkpoint.pt"
    save_checkpoint(checkpoint_filename, model, predictor, optimizer, args, elapsed_time, test_metrics)
