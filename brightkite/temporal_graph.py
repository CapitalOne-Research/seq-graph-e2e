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
import gc
import copy
import pandas as pd
import numpy as np
from torcheval.metrics import ReciprocalRank, HitRate
from datetime import datetime

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
    
    parser.add_argument('--dataset', type=str, default='brightkite',
                        help="Base path for CSV files.")
    # parser.add_argument('--base_path', type=str, default='.',
    #                     help="Base path for CSV files.")
    parser.add_argument('--model', type=str, choices=['TGN', 'DyRep','TNCN'], default='TGN',
                        help="Model type to use.")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help="Dimensionality of input node embeddings.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument('--train_batch_size', type=int, default=2**12,
                        help="Mini-batch size for training.")
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
    parser.add_argument('--train_neg_sampling_ratio', type=float, default=5,
                        help="Negative sampling ratio for training in the neighbor loader.")
    parser.add_argument('--hits_k', type=str, default="1,3,5,10,20,50,100",
                        help="Comma separated list of k values for computing Hits@k metrics.")
    parser.add_argument('--experiment_name', type=str, default='TG-exp-1',
                        help="Experiment name for saving checkpoints and artifacts.")
    parser.add_argument('--friendship_time_location', type=str, default='random',
                        help="Where to place non-timed friendship in the event sequence [last, first, random].")
    parser.add_argument("--max_events", type=int, default=100,
                        help="Maximum number of events per user (will truncate/pad accordingly)")
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
def load_data(base_path, embedding_dim, friendship_time_location, max_events):
    """
    Load edge lists from CSV files with a 'relational_' prefix and create a PyG TemporalData object.
    
    The files are expected to be:
      - relational_train.csv
      - relational_val.csv
      - relational_test.csv
      - relational_val_negative_sample_100.csv
      - relational_test_negative_sample_100.csv
      - personal.csv
      
    Args:
        base_path (str): Base directory where the CSV files are located.
        embedding_dim (int): Dimensionality of node embeddings.
        friendship_time_location (str): Where to place non-timed friendship in the event sequence.
        
    Returns:
        dict: A dictionary containing TemporalData objects for train, val, and test splits,
              and the total number of nodes (num_all_nodes).
    """
    
    logging.info("Preparing dataset...")
    logging.info(f"base path: {base_path}")
    
    train_path = os.path.join(base_path, "relational_train.csv")
    val_path = os.path.join(base_path, "relational_val.csv")
    test_path = os.path.join(base_path, "relational_test.csv")
    val_neg_path = os.path.join(base_path, "relational_val_negative_sample_100.csv")
    test_neg_path = os.path.join(base_path, "relational_test_negative_sample_100.csv")
    personal_path = os.path.join(base_path, "personal.csv")
    
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
    
    src_p = torch.tensor(personal_df['uid'].values, dtype=torch.long)
    dst_p = torch.tensor(codes, dtype=torch.long)
    t_p = torch.tensor(personal_df['timestamp'].values, dtype=torch.long)
    
    if friendship_time_location == 'last':
        src = torch.cat([src_p, src_r], dim=0)
        dst = torch.cat([dst_p, dst_r], dim=0)
        
        next_t = t_p.max() + 100
        t_r = torch.ones(len(src_r), dtype=torch.long) * next_t
        t = torch.cat([t_p, t_r], dim=0)
        
    elif friendship_time_location == 'first':
        src = torch.cat([src_r, src_p], dim=0)
        dst = torch.cat([dst_r, dst_p], dim=0)
        
        next_t = t_p.min() - 100
        t_r = torch.ones(len(src_r), dtype=torch.long) * next_t
        t = torch.cat([t_r, t_p], dim=0)
        
    elif friendship_time_location == 'random':
        src = torch.cat([src_p, src_r], dim=0)
        dst = torch.cat([dst_p, dst_r], dim=0)
        
        next_t = t_p.max() + 100
        t_r = torch.randint(t_p.min() - 100, t_p.max() + 100, (len(src_r),))
        t = torch.cat([t_p, t_r], dim=0)
        
        # reorder by time
        indices = torch.argsort(t)
        src = src[indices]
        dst = dst[indices]
        t   = t[indices]
        
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
    src_test_pos = torch.tensor(val_df['uid'].values, dtype=torch.long)
    dst_test_pos = torch.tensor(val_df['other_uid'].values, dtype=torch.long)
    t_test_pos = torch.ones(len(src_test_pos), dtype=torch.long) * (t_p.max() + 1000)
    
    src_test_neg = torch.tensor(val_neg_df['uid'].values, dtype=torch.long)
    dst_test_neg = torch.tensor(val_neg_df['other_uid'].values, dtype=torch.long)
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

def test(model, data, eval_batch_size, device, hits_k, split="test"):
    """
    Evaluate model performance on a given split using torcheval.metrics.
    
    Args:
        model (dict): Dictionary containing the model modules.
        data (dict): Dictionary containing TemporalData objects.
        eval_batch_size (int): Mini-batch size for evaluation.
        device (torch.device): Device to run computations on.
        hits_k (list): List of k values for computing Hits@k.
        split (str): Which split to evaluate on; "test" or "val".
    
    Returns:
        dict: A dictionary with evaluation metrics.
    """
    embedding_module = model['embedding_module']
    memory = model['memory']
    gnn = model['gnn']
    linkpred = model['linkpred']
    neighbor_loader = model['neighbor_loader']
    assoc = model['assoc']
    assoc2 = model['assoc2']
    
    embedding_module.eval()
    memory.eval()
    gnn.eval()
    linkpred.eval()
    
    pos_scores_list = []
    neg_scores_list = []
    
    train_data = data['train_data']
    
    if split == "val":
        desc_prefix = "Validation"
        pos_data = data['val_data_pos']
        neg_data = data['val_data_neg']
        
        logging.info("Starting evaluation...")
    else:
        desc_prefix = "Testing"
        pos_data = data['test_data_pos']
        neg_data = data['test_data_neg']
        
        logging.info("Starting testing...")
    
    pos_loader = TemporalDataLoader(pos_data, batch_size=eval_batch_size, neg_sampling_ratio=0)
    neg_loader = TemporalDataLoader(neg_data, batch_size=eval_batch_size, neg_sampling_ratio=0)
    
    with torch.no_grad():
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
            
            if args.model == 'TNCN':
                time_info = (last_update, batch.t)                
                scores = linkpred(z, edge_index, torch.stack([assoc[batch.src], assoc[batch.dst]]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info)
            else:
                scores = linkpred(z[assoc[batch.src]], z[assoc[batch.dst]])
            
            pos_scores_list.append(scores.cpu())
            
        for batch in tqdm(neg_loader, desc=f"{desc_prefix} Neg", leave=False):
            batch = batch.to(device)
            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            
            neigh_nid = torch.unique(torch.cat([batch.src, batch.dst, edge_index[0]]))
            assoc2[neigh_nid] = torch.arange(neigh_nid.size(0), device=device)
            
            # get emb for gnn
            x = embedding_module(neigh_nid)
            msg = torch.cat([x[assoc2[edge_index[0]]], x[assoc2[edge_index[1]]]], dim=1)
            
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, train_data.t[e_id.cpu()].to(device), msg)
            
            if args.model == 'TNCN':
                time_info = (last_update, batch.t)                
                scores = linkpred(z, edge_index, torch.stack([assoc[batch.src], assoc[batch.dst]]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info)
            else:
                scores = linkpred(z[assoc[batch.src]], z[assoc[batch.dst]])
            
            neg_scores_list.append(scores.cpu())
    
    pos_scores = torch.cat(pos_scores_list, dim=0)
    neg_scores = torch.cat(neg_scores_list, dim=0)
    
    # note that the positive and negative samples are in the right order in the data frame
    # this pairing of pos and neg sample are valid
    num_samples = pos_scores.shape[0]
    num_neg = neg_scores.shape[0] // num_samples
    neg_scores = neg_scores.reshape(num_samples, num_neg)
    
    candidates = torch.cat([pos_scores, neg_scores], dim=1)
    target = torch.zeros(num_samples, dtype=torch.long) # positive sample is always in 0 idx
    
    metrics = evaluate(candidates, target, k_list=hits_k)
    return metrics

def train(model, data, optimizer, criterion, epochs, train_batch_size, eval_batch_size,
          device, patience, eval_every, enable_early_stopping, train_neg_sampling_ratio, hits_k):
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
        dict: Updated model dictionary after training.
    """
    train_data = data['train_data']
    train_loader = TemporalDataLoader(train_data, batch_size=train_batch_size, neg_sampling_ratio=train_neg_sampling_ratio)
    
    embedding_module = model['embedding_module']
    memory = model['memory']
    gnn = model['gnn']
    linkpred = model['linkpred']
    neighbor_loader = model['neighbor_loader']
    assoc = model['assoc']
    assoc2 = model['assoc2']
    
    embedding_module.train()
    memory.train()
    gnn.train()
    linkpred.train()
    
    best_val_mrr = -float('inf')
    patience_counter = 0
    best_state = None

    logging.info("Starting training...")
    
    last_epoch_time = None
    last_epoch_timestamp = None

    for epoch in range(1, epochs + 1):
        # record epoch start time and ISO timestamp
        epoch_start_time = time.time()
        epoch_timestamp = datetime.now().isoformat()

        memory.reset_state()
        neighbor_loader.reset_state()
        
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
                                    
            # load neighborhood
            n_id, edge_index, e_id = neighbor_loader(batch.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            
            neigh_nid = torch.unique(torch.cat([batch.src, batch.dst, edge_index[0], edge_index[1]]))
            assoc2[neigh_nid] = torch.arange(neigh_nid.size(0), device=device)
            
            # get emb for gnn
            x = embedding_module(neigh_nid)
            msg = torch.cat([x[assoc2[edge_index[0]]], x[assoc2[edge_index[1]]]], dim=1)
            
            # Get updated memory of all nodes involved in the computation.
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, train_data.t[e_id.cpu()].to(device), msg)
            
            if args.model == 'TNCN':
                time_info = (last_update, batch.t)                
                pos_out = linkpred(z, edge_index, torch.stack([assoc[batch.src], assoc[batch.dst]]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info)
                neg_src = batch.src.repeat_interleave(train_neg_sampling_ratio, dim=0)
                neg_out = linkpred(z, edge_index, torch.stack([assoc[neg_src], assoc[batch.neg_dst]]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info)
            else:
                pos_out = linkpred(z[assoc[batch.src]], z[assoc[batch.dst]])
                neg_src = batch.src.repeat_interleave(train_neg_sampling_ratio, dim=0)
                neg_out = linkpred(z[assoc[neg_src]], z[assoc[batch.neg_dst]])
                
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            # Update memory and neighbor loader with ground-truth state.
            msg_gt = torch.cat([x[assoc2[batch.src]], x[assoc2[batch.dst]]], dim=1).detach()
            memory.update_state(batch.src, batch.dst, batch.t, msg_gt)
            neighbor_loader.insert(batch.src, batch.dst)            

            loss.backward()
            optimizer.step()
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
                    best_state = {k: copy.deepcopy(v.state_dict()) for k, v in model.items()
                                  if hasattr(v, "load_state_dict")}
                    logging.info(f"Logged these model params: {best_state.keys()}")
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
if __name__ == "__main__":
    args = parse_args()
    args.base_path = f'../data/tasks/{args.dataset}/friend_recommendation/'
    # args.base_path = f'/work/hdd/bdeb/chenyuen0103/event-seq-graph/data/tasks/{args.dataset}/friend_recommendation/'
    # Process the custom hits_k argument as a list of integers.
    args.hits_k = [int(x) for x in args.hits_k.split(",")]
    
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    project_name = f'temporal-results-{args.dataset}'
    run_name = f"{args.model.lower()}_lr{args.lr}_bs{args.train_batch_size}_nngb{args.num_neighbors}_epochs{args.epochs}_seed{args.seed}"
    logger = LocalLogger(project_name=project_name, run_name=run_name)
    
    if logger.skip:
        exit()
    data = load_data(args.base_path, args.embedding_dim, args.friendship_time_location, args.max_events)
    num_all_nodes = data['num_all_nodes']
    
    # model
    embedding_dim = args.embedding_dim
    msg_dim = memory_dim = time_dim = 2 * args.embedding_dim
    
    embedding_module = torch.nn.Embedding(num_all_nodes, embedding_dim).to(device)
    
    if args.model == 'TGN' or args.model == 'TNCN':
        memory = TGNMemory(
            num_all_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)
    else:
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
        NCN_MODE = 1
        CN_TIME_DECAY = False
        linkpred = NCNPredictor(in_channels=embedding_dim, hidden_channels=embedding_dim,
                            out_channels=1, NCN_mode=NCN_MODE).to(device)
    else:
        linkpred = LinkPredictor(in_channels=embedding_dim).to(device)
    neighbor_loader = LastNeighborLoader(num_all_nodes, size=args.num_neighbors, device=device)
    
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(num_all_nodes, dtype=torch.long, device=device)
    assoc2 = torch.empty(num_all_nodes, dtype=torch.long, device=device)
    
    model = {
        'embedding_module': embedding_module,
        'memory': memory,
        'gnn': gnn,
        'linkpred': linkpred,
        'neighbor_loader': neighbor_loader,
        'assoc': assoc,
        'assoc2': assoc2
    }
    
    optimizer = torch.optim.Adam(
        set(memory.parameters()) |
        set(gnn.parameters()) |
        set(linkpred.parameters()) |
        set(embedding_module.parameters()),
        lr=args.lr
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    start_time = time.time()
    model = train(
        model, data, optimizer, criterion, args.epochs,
        args.train_batch_size, args.eval_batch_size, device,
        patience=args.patience, eval_every=args.eval_every,
        enable_early_stopping=args.enable_early_stopping,
        train_neg_sampling_ratio=args.train_neg_sampling_ratio,
        hits_k=args.hits_k
    )
    elapsed_time = time.time() - start_time

    test_metrics = test(model, data, args.eval_batch_size, device, args.hits_k, split="test")
    log_msg = f"{args.model}: MRR: {test_metrics['MRR']:.4f}"
    for k in args.hits_k:
        log_msg += f", Hits@{k}: {test_metrics[f'Hits@{k}']:.4f}"
    log_msg += f", Time: {elapsed_time:.2f}s"
    logging.info(log_msg)
    final_log_payload = {
        'summary': True,
        'selection': 'validation_mrr' if args.enable_early_stopping else 'final_epoch',
        'elapsed_time': elapsed_time,
        **{f"test_{k.lower().replace('@', '_at_')}": v for k, v in test_metrics.items()}
    }
    logger.log(final_log_payload)
    logger.finish()
    os.makedirs("stored", exist_ok=True)
    checkpoint_filename = f"stored/{args.experiment_name}_checkpoint.pt"
    save_checkpoint(checkpoint_filename, model, optimizer, args, elapsed_time, test_metrics)
