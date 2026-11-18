import argparse
import logging
import pickle
import random
import json
import os
import sys
import tempfile
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer, BertSelfAttention
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv, TransformerConv
# from gcn_layer import GCNConv
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from torcheval.metrics import ReciprocalRank, HitRate
import random # <-- Required for the new lookup method
from sequence_embedding import *
import time
from torch.nn.utils.rnn import pad_sequence
from xat import AttentionCompressor
from xat import TokenCrossAttnConv #, TokenCrossAttnConvGCN
import itertools
from torch_geometric.nn import aggr
from torch.nn.parallel import DistributedDataParallel as DDP # Add for DDP handling
from collections import OrderedDict
from torch.nn import Parameter, ModuleList, MultiheadAttention
import copy
import re
import math
from typing import Optional, Dict, Any, Literal
# Better: combine with an environment variable
# api_key = os.getenv("WANDB_API_KEY")
# if api_key:
#     wandb.login(key=api_key)



# -----------------------------------------------------------------------------
# Configure Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def prefix_metrics(d, prefix="test_"):
    out = {}
    for k, v in d.items():
        # make keys CSV/regex-friendly: lowercase and replace '@'
        kk = k.lower().replace('@', '_at_')
        out[f"{prefix}{kk}"] = float(v)
    return out
def append_row_to_json_log(json_path: str, row: dict):
    """Append `row` to a JSON log list (create if missing)."""
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                log = json.load(f)
            if not isinstance(log, list):
                log = [log]
        else:
            log = []
    except Exception:
        log = []
    log.append(row)
    with open(json_path, "w") as f:
        json.dump(log, f, indent=4)

def _results_json_has_test_metrics(path: str) -> bool:
    """
    Returns True if `path` looks like a completed run output.
    We consider it completed if any row contains test metrics (e.g., 'test_mrr').
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return False
        for row in data:
            if not isinstance(row, dict):
                continue
            if "test_mrr" in row or "best_test_mrr" in row:
                return True
    except Exception:
        return False
    return False

def parse_compress_dim_from_name(filename):
    """
    Extracts the compress_dim from a filename string, e.g.,
    '..._compress-dim2_...' -> returns 2
    """
    m = re.search(r'compress-dim(\d+)_', filename)
    if m:
        return int(m.group(1))
    return None


def set_seed(seed: int):
    """
    Sets the seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed)  # PyG's comprehensive seeder

    # Ensure deterministic behavior on CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Global seed set to {seed}")


# -----------------------------------------------------------------------------
# 2. Combined Model Architecture
# -----------------------------------------------------------------------------


def append_row_to_json_log(json_path: str, row: dict):
    """Append `row` to a JSON log list (create if missing)."""
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                log = json.load(f)
            if not isinstance(log, list):
                log = [log]
        else:
            log = []
    except Exception:
        log = []
    log.append(row)
    with open(json_path, "w") as f:
        json.dump(log, f, indent=4)


class SequenceEncoder(nn.Module):
    """The sequence model part (BERT) to encode user event sequences."""
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask):
        # We use the [CLS] token's final hidden state as the sequence embedding.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs.last_hidden_state[:, -1, :] # (batch_size, hidden_size)



class SequenceEncoder_mean(nn.Module):
    """Encodes sequences and computes the average embedding efficiently."""
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask):
        # 1. Get token embeddings from the model
        # shape: (batch_size, sequence_length, hidden_size)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Zero out the embeddings of padding tokens
        # The mask is expanded via broadcasting, which is memory-efficient
        mask = attention_mask.unsqueeze(-1)
        last_hidden_state = last_hidden_state * mask

        # 3. Sum the embeddings and divide by the number of real tokens
        sum_embeddings = torch.sum(last_hidden_state, dim=1)
        
        # Get the number of actual tokens by summing the attention mask
        # We unsqueeze to make the division broadcast correctly
        num_tokens = torch.sum(attention_mask, dim=1).unsqueeze(-1)

        # Clamp to avoid division by zero for empty sequences
        return sum_embeddings / num_tokens.clamp(min=1e-9)


class SequenceEncoder_4d(nn.Module):
    """
    Encodes sequences using a BERT model and compresses the output into a fixed-size
    representation of 4 vectors corresponding to mean, max, min, and last token pooling.
    This mimics the behavior of a fixed compressor with k=4.
    """
    def __init__(self, config):
        """
        Initializes the encoder with a BERT model.

        Args:
            config: A Hugging Face BertConfig object.
        """
        super().__init__()
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            input_ids (torch.Tensor): Tensor of input token ids.
                                      Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Tensor of attention masks (1 for real tokens, 0 for padding).
                                           Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: The compressed sequence representation.
                          Shape: (batch_size, 4, hidden_size)
        """
        # 1. Get token embeddings from the BERT model
        # Shape: (batch_size, sequence_length, hidden_size)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Prepare a broadcastable mask for element-wise operations
        # Shape: (batch_size, sequence_length, 1)
        expanded_mask = attention_mask.unsqueeze(-1)

        # --- Mean Pooling ---
        # Zero out the embeddings of padding tokens
        masked_for_mean = last_hidden_state * expanded_mask
        # Sum the embeddings and divide by the number of real tokens
        sum_embeddings = torch.sum(masked_for_mean, dim=1)
        num_tokens = torch.sum(attention_mask, dim=1).unsqueeze(-1)
        # Clamp to avoid division by zero for empty sequences
        mean_pooled = sum_embeddings / num_tokens.clamp(min=1e-9)

        # --- Max Pooling ---
        # To correctly ignore padding, set padded token embeddings to a very small value
        masked_for_max = last_hidden_state.masked_fill(~expanded_mask.bool(), -torch.finfo(last_hidden_state.dtype).max)
        max_pooled = torch.max(masked_for_max, dim=1).values

        # --- Min Pooling ---
        # To correctly ignore padding, set padded token embeddings to a very large value
        masked_for_min = last_hidden_state.masked_fill(~expanded_mask.bool(), torch.finfo(last_hidden_state.dtype).max)
        min_pooled = torch.min(masked_for_min, dim=1).values

        # --- Last Token Pooling ---
        # Find the index of the last non-padded token for each sequence in the batch
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_size = input_ids.shape[0]
        # Create a tensor of batch indices [0, 1, 2, ...] to use for advanced indexing
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        # Gather the embeddings of the last token for each sequence
        last_token_embeddings = last_hidden_state[batch_indices, sequence_lengths]

        # 4. Stack the pooled outputs
        # Each pooled tensor has shape (batch_size, hidden_size).
        # We unsqueeze them to (batch_size, 1, hidden_size) before concatenating.
        output = torch.cat(
            [mean_pooled, max_pooled, min_pooled, last_token_embeddings],
            dim=1
        )

        # The final output shape is (batch_size, 4, hidden_size)
        return output


class SequenceEncoder_Stack(nn.Module):
    """The sequence model part (BERT) to encode user event sequences."""
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # breakpoint()
        return outputs.last_hidden_state # (batch_size, hidden_size)


# --- The Updated CrossAttentionLayer ---
class CrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attention = BertSelfAttention(config)
        self.output_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, query_states, context_states, attention_mask=None):
        attention_output = self.cross_attention(
            hidden_states=query_states, 
            encoder_hidden_states=context_states,
            attention_mask=attention_mask
        )[0]
        dense_output = self.output_dense(attention_output)
        dropout_output = self.dropout(dense_output)
        normalized_output = self.LayerNorm(dropout_output + query_states)
        return normalized_output.squeeze(1)
    
    




class GraphModel(nn.Module):
    """
    A flexible GNN model that can stack different layer types dynamically.
    """
    def __init__(self, in_channels, hidden_channels, gnn_layer_types: list[str],
                 dropout, heads, k_aggregations, args=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        
        current_in_channels = in_channels
        gnn_layer_types = gnn_layer_types.split(",")
        for i, model_type in enumerate(gnn_layer_types):
            if model_type == 'TokenXAT':
                # The custom TokenCrossAttnConv layer
                layer = TokenCrossAttnConv(
                    in_channels=current_in_channels,
                    # out_channels=hidden_channels, # Note: out_channels is per-head
                    out_channels = hidden_channels // heads,
                    heads=heads,
                    num_agg_queries = args.compress_dim,
                    pre_agg_proj = args.pre_agg_proj,
                    post_agg_proj = args.post_agg_proj,
                    aggregation_mode = args.aggregation_mode,
                    add_self_loops = args.add_self_loops,
                    use_qk_norm = args.qk_norm,
                    compress_first = args.compress_first,
                    message_mode = args.msg_mode,
                    gcn_edge_weight = args.gcn_edge_weight,
                    use_layer_norm = args.layer_norm
                )

                current_in_channels = layer.k * hidden_channels

            elif model_type == 'GCN':
                # breakpoint()
                layer = GCNConv(current_in_channels, current_in_channels)
            elif model_type == 'GraphSAGE':
                layer = SAGEConv(current_in_channels, current_in_channels)
                # current_in_channels = hidden_channels
            elif model_type == 'GAT':
                # GAT's output is concatenated, so it becomes hidden_channels
                layer = GATConv(current_in_channels,  current_in_channels// heads, heads=heads, dropout=dropout, residual = True)
                # current_in_channels = hidden_channels     
            elif model_type == 'GATv2':
                # GAT's output is concatenated, so it becomes hidden_channels
                layer = GATv2Conv(current_in_channels,  current_in_channels// heads, heads=heads, dropout=dropout, residual = True)
                # current_in_channels = hidden_channels  
            elif model_type == 'TransformerConv':
                # GAT's output is concatenated, so it becomes hidden_channels
                layer = TransformerConv(current_in_channels,  current_in_channels// heads, heads=heads, dropout=dropout)
                # current_in_channels = hidden_channels    
            else:
                raise ValueError(f"Unsupported GNN layer type: {model_type}")

            self.layers.append(layer)

    def forward(self, x, edge_index, user_timestamps=None, pad_mask=None, training = True):
        # breakpoint()
        for i, layer in enumerate(self.layers):
            is_last_layer = (i == len(self.layers) - 1)
            if isinstance(layer, TokenCrossAttnConv):
                x = layer(x, edge_index, mask=pad_mask, training = training)
            else:
                if x.dim() > 2:
                    x = x.view(x.shape[0], -1)
                x = layer(x, edge_index)

            # Apply activation and dropout between layers
            if not is_last_layer:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
            if is_last_layer:
                if x.dim() > 2:
                    # Flatten the output from the previous (TokenXAT) layer
                    x = x.view(x.shape[0], -1)
        
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
        if z_src.dim() == 3:
            z_src = z_src.view(z_src.size(0), -1)
            z_dst = z_dst.view(z_dst.size(0), -1)
        
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)
    



class JointModel(nn.Module):
    """The main model that combines the sequence encoder and the graph model."""
    def __init__(self, bert_config, gnn_in_channels, gnn_hidden_channels, gnn_num_layers, gnn_dropout, gnn_heads, args = None):
        super().__init__()
        # self.aggregations = ['mean', 'max', 'min']
        self.aggregations = ['mean']
        # breakpoint()
        self.aggregation_mode = args.aggregation_mode
        self.k = 1
        if args.gnn_layer_types.split(",")[0] in ['TokenXAT','TokenXLinAT']:
            self.sequence_encoder = SequenceEncoder_Stack(bert_config)
            self.k = args.compress_dim
        else:
            if args.encoder_compress == 'mean':
                self.sequence_encoder = SequenceEncoder_mean(bert_config)
            elif args.encoder_compress == '4d':
                self.sequence_encoder = SequenceEncoder_4d(bert_config)
                gnn_in_channels *= 4
                gnn_hidden_channels *= 4
            else:
                self.sequence_encoder = SequenceEncoder(bert_config)
        self.heads = gnn_heads
        self.out_channels = gnn_hidden_channels // self.heads

        self.gnn = GraphModel(
            in_channels=gnn_in_channels,
            hidden_channels=gnn_hidden_channels,
            gnn_layer_types=args.gnn_layer_types, # Pass the new config string
            dropout=gnn_dropout,
            heads=gnn_heads,
            k_aggregations=self.k, # Pass k for shape calculation
            args=args,
        )
        

        # --- Initialize Aggregator for Residual Connection ---
        if args.gnn_layer_types.split(",")[0] in ['TokenXAT']:
            self.residual_compressor = self.gnn.layers[0].compressor

        # 2. Add a projection layer to get the final desired dimension 'd'
        # The input is k * gnn_hidden_channels
        # self.proj = nn.Linear(self.k * gnn_hidden_channels, gnn_hidden_channels)
        
        self.predictor = LinkPredictor(gnn_hidden_channels * self.k)
        # self.projector = AttentionProjector(args.max_events * 4 + 1)



    def forward(self, user_sequences_ids, user_sequences_masks, graph_data, user_timestamps = None, training = True):
        """
        A forward pass for joint training.
        
        Returns:
            mlm_logits (for sequence loss)
            link_preds (for graph loss)
        """
        # 1. Sequence Model Forward Pass
        # First, get embeddings for all users present in the current graph batch
        all_user_ids_in_batch = graph_data.n_id
        
        # Select the sequences and masks for the users in the current computational graph
        batch_seq_ids = user_sequences_ids[all_user_ids_in_batch]
        batch_seq_masks = user_sequences_masks[all_user_ids_in_batch]
        batch_seq_times = user_timestamps[all_user_ids_in_batch] if user_timestamps.any() else None
        # print("Sequence IDs (first example):", batch_seq_ids[0])
        # print("Padding Mask (first example):", batch_seq_masks[0])
        # breakpoint()
        # Get user embeddings from the sequence model
        user_embeddings = self.sequence_encoder(batch_seq_ids, batch_seq_masks)
        node_features_after_gnn = self.gnn(user_embeddings, graph_data.edge_index, batch_seq_times, pad_mask = batch_seq_masks,training = training)
        
        # breakpoint()


        if user_embeddings.dim() == 3:
            # N, T, d = user_embeddings.shape
            # tokens_reshaped = user_embeddings.view(-1, d) 
            # non_pad_mask = batch_seq_masks.bool()
            # valid_tokens = user_embeddings[non_pad_mask]
            # token_index = torch.arange(N, device=user_embeddings.device).repeat_interleave(T)
            # valid_token_index = token_index[non_pad_mask.view(-1)]
            # aggregated_tokens = self.token_aggr_module(valid_tokens, valid_token_index, dim_size = N)
            # # aggregated_tokens = self.token_aggr_module(tokens_reshaped, token_index, dim_size=N)
            N, T, d = user_embeddings.shape
            non_pad_mask = batch_seq_masks.bool()
            valid_tokens = user_embeddings[non_pad_mask]
            token_index = torch.arange(N, device=user_embeddings.device).repeat_interleave(T)
            valid_token_index = token_index[non_pad_mask.view(-1)]

            grouped_tokens, key_padding_mask = to_dense_batch(
                valid_tokens, valid_token_index, batch_size=N
            )
            # Call the compressor to get the aggregated residual features
            residual_features = self.residual_compressor(grouped_tokens, key_padding_mask=key_padding_mask,training=self.training)
            residual_features = residual_features.reshape(N, -1)
                
        else:
            residual_features = user_embeddings

        combined_features = residual_features + node_features_after_gnn
        

        return combined_features


def load_and_prepare_data(args):
    """Loads and preprocesses all data, handling subsetting if requested."""
    logger.info("Starting data loading and preparation...")
    
    # --- Subsetting Logic ---
    train_df = pd.read_csv(os.path.join(args.input_path, "relational_train.csv"), nrows=args.debug_subset)
    seq_df = pd.read_csv(args.csv_file)
    val_df = pd.read_csv(os.path.join(args.input_path, "relational_val.csv"))
    if args.num_negative_samples < 1000:
        val_neg_df = pd.read_csv(os.path.join(args.input_path, "relational_val_negative_sample_100.csv"))
    else:
        val_neg_df = pd.read_csv(os.path.join(args.input_path, "relational_val_negative_sample.csv"))
    if args.debug_subset and args.debug_subset > 0:
        logger.info(f"--- Running in DEBUG mode: using a subset of {args.debug_subset} training interactions ---")
        # Load a subset of training graph edges
        # Identify the active nodes in this subset
        active_nodes = sorted(list(pd.unique(train_df[['uid', 'other_uid']].values.ravel('K'))))
        # Create a mapping from original node IDs to new, dense IDs (0 to N-1)
        node_map = {old_id: new_id for new_id, old_id in enumerate(active_nodes)}
        logger.info(f"Subset has {len(active_nodes)} unique users.")

        # Remap the training data
        train_df['uid'] = train_df['uid'].map(node_map)
        train_df['other_uid'] = train_df['other_uid'].map(node_map)
        
        # Load and filter sequence data
        
        seq_df = seq_df[seq_df['uid'].isin(active_nodes)]
        seq_df['uid'] = seq_df['uid'].map(node_map)

        # Load, filter, and remap validation data
        val_df = val_df[val_df['uid'].isin(active_nodes) & val_df['other_uid'].isin(active_nodes)]
        val_df = val_df.head(args.debug_subset) # Truncate to debug size
        val_df['uid'] = val_df['uid'].map(node_map)
        val_df['other_uid'] = val_df['other_uid'].map(node_map)

        
        val_neg_df = val_neg_df[val_neg_df['uid'].isin(active_nodes) & val_neg_df['other_uid'].isin(active_nodes)]
        val_neg_df = val_neg_df.head(args.debug_subset) # Truncate to debug size
        val_neg_df['uid'] = val_neg_df['uid'].map(node_map)
        val_neg_df['other_uid'] = val_neg_df['other_uid'].map(node_map)
        
        num_nodes = len(active_nodes)

    else:
        logger.info("--- Running in normal mode: using the full dataset ---")

        num_nodes = int(pd.concat([train_df['uid'], train_df['other_uid']]).max()) + 1

    # --- Common Data Processing ---
    # Keep train link labels separate from the message-passing graph so
    # no-relational ablations can still optimize on the same supervision.
    train_edge_index = torch.tensor(train_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    train_edge_label_index = T.ToUndirected()(
        Data(edge_index=train_edge_index, num_nodes=num_nodes)
    ).edge_index

    # Optionally remove all relational edges from the message-passing graph
    # while keeping the same train/validation/test link-prediction labels.
    if args.include_relational:
        data = Data(edge_index=train_edge_index, num_nodes=num_nodes)
        data = T.ToUndirected()(data)
        logger.info("Using relational edges for graph message passing.")
    else:
        data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)
        logger.info("Relational edges disabled: graph message passing uses an empty edge set.")
    data.train_edge_label_index = train_edge_label_index
    
    # Add validation edges to data object
    data.val_edge_index = torch.tensor(val_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    data.val_neg_edge_index = torch.tensor(val_neg_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    logger.info(f"Graph loaded: {data}")
    
    # # --- ADD THIS PART ---
    # # Load and add test edges to the data object
    logger.info("Loading test set data...")
    test_df = pd.read_csv(os.path.join(args.input_path, "relational_test.csv"))
    test_neg_df = pd.read_csv(os.path.join(args.input_path, "relational_test_negative_sample_100.csv"))
    
    # # Handle subsetting for debug mode if necessary
    if args.debug_subset and args.debug_subset > 0:
        active_nodes = set(node_map.keys())
        test_df = test_df[test_df['uid'].isin(active_nodes) & test_df['other_uid'].isin(active_nodes)]
        test_df['uid'] = test_df['uid'].map(node_map)
        test_df['other_uid'] = test_df['other_uid'].map(node_map)
        
        test_neg_df = test_neg_df[test_neg_df['uid'].isin(active_nodes) & test_neg_df['other_uid'].isin(active_nodes)]
        test_neg_df['uid'] = test_neg_df['uid'].map(node_map)
        test_neg_df['other_uid'] = test_neg_df['other_uid'].map(node_map)


    data.test_edge_index = torch.tensor(test_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    data.test_neg_edge_index = torch.tensor(test_neg_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    # # --- END OF ADDED PART ---

    logger.info(f"Graph loaded: {data}")
    
    # Process sequences
    if args.event_set:
        seq_df = seq_df[seq_df["event_set"] == args.event_set]
    seq_df = seq_df.sort_values(by=["uid", "timestamp"])
    # user_groups = seq_df.groupby("uid")["event"].apply(list)
    user_groups = seq_df.groupby("uid")[['event', 'timestamp']].apply(
        lambda g: list(zip(g['event'], g['timestamp']))
    )
    uids = user_groups.index.tolist()
    user_sequences = {uid: seq[-args.max_events:] for uid, seq in user_groups.items()}
    logger.info(f"Loaded sequences for {len(user_sequences)} users.")
    return data, uids, user_sequences

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e6:
        return f'{num_params / 1e6:.2f}M'
    else:
        return f'{num_params / 1e3:.2f}K'
    
    



def save_checkpoint(filepath, model, optimizer, args, elapsed_time, test_metrics):
    """
    Saves the JointModel, optimizer, and training metadata to a checkpoint file.
    """
    # If the model is wrapped in DDP, we save the underlying model's state_dict
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    
    checkpoint = {
        'experiment_name': args.experiment_name,
        'model_state_dict': model_state, # Contains all sub-modules
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'elapsed_time': elapsed_time,
        'test_metrics': test_metrics,
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved at {filepath}")



def load_checkpoint(filepath, model, optimizer, device):
    """
    Loads a checkpoint into the model and optimizer.
    """
    try:
        checkpoint = torch.load(filepath, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # If the checkpoint was saved from a DDP model, it will have a 'module.' prefix
        # on all keys. We need to remove it to load into a non-DDP model.
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logging.info(f"Checkpoint loaded successfully from {filepath}")
        return checkpoint.get('args', {}), checkpoint.get('test_metrics', {})
        
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {filepath}. Starting from scratch.")
        return None, None
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}. Starting from scratch.")
        return None, None

logger = logging.getLogger(__name__)


def safe_load_checkpoint(model, ckpt_path, device):
    """
    Load only the keys from the checkpoint that exist in the current model
    and have matching shapes.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)  # handle wrapped dicts
    
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped.append((k, v.shape, model_state[k].shape))
    
    # Actually load
    msg = model.load_state_dict(filtered_state, strict=False)
    
    print(f"[safe_load] Loaded {len(filtered_state)} matching tensors.")
    if skipped:
        print(f"[safe_load] Skipped {len(skipped)} due to shape mismatch:")
        for name, ckpt_shape, model_shape in skipped[:10]:
            print(f"  - {name}: ckpt={ckpt_shape}, model={model_shape}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")
    
    if msg.missing_keys:
        print(f"[safe_load] Missing keys in checkpoint: {len(msg.missing_keys)}")
        for k in msg.missing_keys[:10]:
            print("  -", k)
    if msg.unexpected_keys:
        print(f"[safe_load] Unexpected keys in checkpoint: {len(msg.unexpected_keys)}")
        for k in msg.unexpected_keys[:10]:
            print("  -", k)
    
    return ckpt

# Assume a logger is configured, e.g., logger = logging.getLogger(__name__)

class BestCheckpointManager:
    """
    Saves 'last' every epoch and overwrites 'best' when a monitored metric improves.
    Uses atomic rename to avoid partial files on crash.
    """
    def __init__(
        self,
        base_checkpoint_path: str,             # e.g., ".../checkpoints/<run>.pth"
        mode: Literal["max","min"] = "max",    # "max" for MRR/AUC; "min" for loss
        greater_is_better: Optional[bool] = None
    ):
        self.base = base_checkpoint_path
        self.last_path = self.base                       # ".../<run>.pth"
        self.best_path = self.base.replace(".pth", "_best.pth")
        if greater_is_better is None:
            greater_is_better = (mode == "max")
        self.greater_is_better = greater_is_better
        self.best_value = -math.inf if self.greater_is_better else math.inf
        # If a best file exists, try to read prior best value from its sidecar json (optional)
        self.best_meta_path = self.best_path.replace(".pth", ".json")

    def _improved(self, new: float) -> bool:
        if self.greater_is_better:
            return new > self.best_value
        else:
            return new < self.best_value

    @staticmethod
    def _atomic_save(obj: Dict[str, Any], path: str):
        # Save to a temp file in the same directory then atomically replace.
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=d) as tmp:
            torch.save(obj, tmp.name)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmppath = tmp.name
        os.replace(tmppath, path)  # atomic on POSIX

    @staticmethod
    def _atomic_write_json(obj: Dict[str, Any], path: str):
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=d) as tmp:
            json.dump(obj, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmppath = tmp.name
        os.replace(tmppath, path)

    def save_last(
        self,
        model,
        optimizer,
        args,
        elapsed_time: float,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        payload = {
            "experiment_name": getattr(args, "experiment_name", ""),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "elapsed_time": elapsed_time,
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "metrics": metrics or {},
        }
        self._atomic_save(payload, self.last_path)
        logging.info(f"[ckpt] Saved last checkpoint -> {self.last_path}")

    def save_if_best(
        self,
        model,
        optimizer,
        args,
        elapsed_time: float,
        epoch: int,
        monitored_value: float,
        metrics: Dict[str, float],
        monitored_name: str = "MRR"
    ) -> bool:
        """Returns True iff a new best was saved."""
        if not self._improved(monitored_value):
            return False
        self.best_value = monitored_value
        payload = {
            "experiment_name": getattr(args, "experiment_name", ""),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "elapsed_time": elapsed_time,
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "metrics": metrics,
            "best_key": monitored_name,
            "best_value": float(monitored_value),
        }
        self._atomic_save(payload, self.best_path)
        self._atomic_write_json(  # sidecar for quick inspection (optional)
            {"best_key": monitored_name, "best_value": float(monitored_value), "epoch": epoch}, 
            self.best_meta_path
        )
        logging.info(f"[ckpt] ✅ New BEST ({monitored_name}={monitored_value:.6f}) -> {self.best_path}")
        return True

def supervised_seq_warmup(model, train_loader, user_sequences_ids, user_sequences_masks,
                          device, warmup_epochs, lr):
    """Supervised-frozen control (variant D): train the sequence encoder on the LINK objective
    WITHOUT graph message passing (sequence-only), so the caller can freeze it and then train the GNN.

    Predicts the relation from paired per-user encoder embeddings via a throwaway link head, reusing
    the joint loader's positive+negative edges. This is TASK-SUPERVISED (not the unsupervised MLM
    ckpt) -- that is what makes it D rather than the (i) frozen-MLM control. Run enough epochs to
    converge (watch the printed loss); a weak encoder makes an E>D gap unconvincing.
    """
    import torch.optim as optim
    warm_pred = None
    warm_opt = None
    for epoch in range(1, warmup_epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for batch in train_loader:
            batch.val_edge_index = None; batch.test_edge_index = None
            batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
            batch = batch.to(device)
            # Sequence-only: encode ONLY the nodes in the supervised edges (no graph / no neighbors).
            # Encoding the whole neighbor-expanded batch through BERT with a trainable encoder OOMs,
            # and the warmup never uses the neighbors anyway.
            uniq, inv = torch.unique(batch.edge_label_index, return_inverse=True)  # inv:(2,E) -> rows of z
            z = model.sequence_encoder(user_sequences_ids[batch.n_id[uniq]], user_sequences_masks[batch.n_id[uniq]])
            if z.dim() == 3:
                z = z.reshape(z.size(0), -1)              # match LinkPredictor's flatten
            if warm_pred is None:                          # lazily size the head to the encoder output
                warm_pred = LinkPredictor(z.size(1)).to(device)
                warm_opt = optim.Adam(
                    list(model.sequence_encoder.parameters()) + list(warm_pred.parameters()), lr=lr)
            warm_opt.zero_grad()
            scores = warm_pred(z[inv[0]], z[inv[1]]).squeeze()
            loss = F.binary_cross_entropy_with_logits(scores, batch.edge_label.float())
            loss.backward()
            warm_opt.step()
            total += loss.item(); nb += 1
        logging.info(f"[supervised-frozen warm-up] epoch {epoch:03d}/{warmup_epochs}  "
                     f"seq-only link BCE: {total / max(1, nb):.4f}")


def train_model(
    # Core components
    model,
    train_loader,
    graph_data,
    user_sequences_ids,
    user_sequences_masks,
    user_timestamps,
    optimizer,
    tokenizer,
    device,
    # Loss function and its specific components
    loss_fn,
    # Training control and hyperparameters
    args, # For args like num_negative_samples
    # Evaluation (optional)
    eval_dataset=None,
    initial_metrics_log=None,
    log_path = None,
    run_name = None,
    ckpt_manager=None,
    start_epoch=1):
    """
    Trains the JointModel for the single task of next-event prediction using
    a contrastive BCE loss. This loop is sequence-driven.
    """
    results_log = initial_metrics_log if initial_metrics_log is not None else []

    # best_val_mrr = results_log[0]['evaluation_metrics']['MRR'] if results_log and not np.isnan(results_log[0]['evaluation_metrics']['MRR']) else -1.0
    best_val_mrr = -1.0
    best_model_state = None
    best_epoch_metrics = None
    best_epoch_elapsed_time = 0.0
    patience_counter = 0
    global_step = 0
    cumulative_elapsed_time = 0.0
    # if args.use_wandb:
    #     wandb.watch(model, log_freq=args.log_interval)

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_begin_time = time.time()
        model.train()
        total_link_loss = 0
        total_mlm_loss = 0
        running_interval_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            global_step += 1
            batch.val_edge_index = None; batch.test_edge_index = None
            batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
            batch = batch.to(device)
            optimizer.zero_grad()
            gnn_out= model(user_sequences_ids, user_sequences_masks, batch, user_timestamps, training = True)
            scores = model.predictor(gnn_out[batch.edge_label_index[0]], gnn_out[batch.edge_label_index[1]]).squeeze()
            link_loss = loss_fn(scores, batch.edge_label.float())
            link_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            
            # --- Accumulate Losses ---
            loss_item = link_loss.item()
            total_link_loss += loss_item
            # if args.use_wandb:
            #     running_interval_loss += loss_item
            
            pbar.set_postfix({'TrainLink': f'{total_link_loss / (i+1):.4f}'})

            
            # --- Log Batch-level Metrics at Intervals ---
            # if args.use_wandb and global_step % args.log_interval == 0:
            #     wandb.log({
            #         'train/batch_loss': running_interval_loss / args.log_interval
            #     }, step=global_step)
            #     running_interval_loss = 0.0 # Reset for next interval

        
                
        avg_epoch_train_loss = total_link_loss/len(train_loader)
        cumulative_elapsed_time += time.time() - epoch_begin_time
        epoch_time = time.time() - epoch_begin_time
        val_loss, val_mrr = np.nan, np.nan
        metrics = None
        eval_time = 0
        
        # ✅ ADDED: GPU Memory Logging
        gpu_mem_allocated = 0
        gpu_mem_peak = 0
        # Only log if CUDA is available and in use
        if device.type == 'cuda':
            # Get current and peak memory in MB
            gpu_mem_allocated = round(torch.cuda.memory_allocated(device) / (1024**2), 2)
            gpu_mem_peak = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            # Reset the peak memory "high-water mark" for the next epoch
            torch.cuda.reset_peak_memory_stats(device)
            
        eval_begin_time = time.time()

        # --- End of Epoch: Validation ---
        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            metrics = test(model, model.predictor, graph_data, args.eval_batch_size, device, args.num_neighbors, args.hits_k, user_sequences_ids, user_sequences_masks, user_timestamps, loss_fn, split="val")
            val_mrr = metrics["MRR"]
            log_msg = f"Epoch {epoch:03d} Evaluation | Val MRR: {val_mrr:.4f}"
            for k in args.hits_k:
                log_msg += f", Hits@{k}: {metrics[f'Hits@{k}']:.4f}"
            logging.info(log_msg)
            
            
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_state = model.state_dict()
                best_epoch_metrics = metrics # <-- Store the metrics dict
                best_epoch_elapsed_time = cumulative_elapsed_time # <-- Store the cumulative time
                patience_counter = 0
                logging.info("New best model found; resetting patience counter.")
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model found at epoch {epoch}")

                # if args.use_wandb:
                #     torch.save(model.state_dict(), "best_model.pth")
                #     artifact = wandb.Artifact(f"{run_name.replace("e2e_", "")})-model", type='model')
                #     artifact.add_file("best_model.pth")
                #     wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}", "best"])
                #     os.remove("best_model.pth")
            else:
                patience_counter += args.eval_every
                logging.info(f"No improvement; patience counter: {patience_counter}/{args.patience}")
                
            eval_time = time.time() - eval_begin_time
            if ckpt_manager is not None and not math.isnan(val_mrr):
                ckpt_manager.save_if_best(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    elapsed_time=cumulative_elapsed_time,
                    epoch=epoch,
                    monitored_value=val_mrr,
                    metrics={"val_MRR": val_mrr, **{f"val_Hits@{k}": float(metrics.get(f"Hits@{k}", float("nan"))) for k in args.hits_k}},
                    monitored_name="MRR"
                )
            if ckpt_manager is not None:
                ckpt_manager.save_last(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    elapsed_time=cumulative_elapsed_time,
                    epoch=epoch,
                    metrics=(metrics or {"train_loss": avg_epoch_train_loss})
                )
            
        # --- Log Epoch Summary to W&B ---
        # if args.use_wandb:
        #     epoch_log_data = {
        #         'epoch': epoch,
        #         'train/epoch_loss': avg_epoch_train_loss,
        #     }
        #     if metrics: # Add validation metrics if they were computed
        #         for key, value in metrics.items():
        #             epoch_log_data[f'val/{key}'] = value
            
        #     wandb.log(epoch_log_data, step=global_step)
            
        epoch_row = {'epoch': epoch, 'train_loss': avg_epoch_train_loss, "evaluation_metrics": metrics,  "epoch_time": epoch_time, "eval_time": eval_time, "gpu_mem_allocated_mb": gpu_mem_allocated,
            "gpu_mem_peak_mb": gpu_mem_peak}
        results_log.append(epoch_row)
        with open(log_path.replace('.csv','.json'), 'w') as f:
                json.dump(results_log, f, indent=4)
        if args.enable_early_stopping:
            if patience_counter >= args.patience:
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
    logger.info("Training complete.")
    return results_log, best_model_state, best_epoch_metrics, best_epoch_elapsed_time
    


# -----------------------------------------------------------------------------
# 5. Evaluation and Main Training Loop
# -----------------------------------------------------------------------------

def evaluate(candidates, target, k_list):
    """Computes retrieval metrics (MRR, Hits@k) from scores."""
    mrr_metric = ReciprocalRank()
    hits_metrics = {k: HitRate(k=k) for k in k_list}
    mrr_metric.update(candidates, target)
    for k, hits_metric in hits_metrics.items():
        hits_metric.update(candidates, target)
    mrr = mrr_metric.compute().mean().item()
    hits_results = {f"Hits@{k}": hits_metrics[k].compute().mean().item() for k in k_list}
    return {"MRR": mrr, **hits_results}

def test(model, predictor, data, eval_batch_size, device, num_neighbors, hits_k, user_sequences_ids, user_sequences_masks, user_timestamps, criterion, split="test", dump_path=None):
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
        # num_neighbors= [num_neighbors] if args.gnn_model == "XAT" else [num_neighbors] * len(model.gnn.layers),
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
        # edge_label_index=subsampled_neg_edge_index,
        batch_size=eval_batch_size,
        shuffle=False,
        neg_sampling_ratio=0
    )
    
    with torch.no_grad():
        for batch in tqdm(pos_loader, desc=f"{desc_prefix} Pos", leave=False):
            batch.val_edge_index = None; batch.test_edge_index = None
            batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
            batch = batch.to(device)
            gnn_out = model(user_sequences_ids, user_sequences_masks, batch, user_timestamps, training = False)
            # breakpoint()
            scores = predictor(gnn_out[batch.edge_label_index[0]], gnn_out[batch.edge_label_index[1]]).squeeze()
            pos_scores_list.append(scores.cpu())
            
        for batch in tqdm(neg_loader, desc=f"{desc_prefix} Neg", leave=False):
            batch.val_edge_index = None; batch.test_edge_index = None
            batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
            batch = batch.to(device)
            gnn_out = model(user_sequences_ids, user_sequences_masks, batch, user_timestamps, training = False)
            scores = predictor(gnn_out[batch.edge_label_index[0]], gnn_out[batch.edge_label_index[1]]).squeeze()
            neg_scores_list.append(scores.cpu())
    
    pos_scores = torch.cat(pos_scores_list, dim=0)
    neg_scores = torch.cat(neg_scores_list, dim=0)
    
    # --- 2. Calculate Loss ---
    # Create labels: 1s for all positive scores, 0s for all negative scores
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    
    # Concatenate all scores and labels
    all_scores = torch.cat([pos_scores, neg_scores], dim=0)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)
    
    # Calculate loss using the provided criterion
    loss = criterion(all_scores, all_labels)
    
    
    # note that the positive and negative samples are in the right order in the data frame
    # this pairing of pos and neg sample are valid
    num_samples = pos_scores.shape[0]
    if num_samples == 0:
        raise RuntimeError("No positive samples to evaluate.")

    remainder = int(neg_scores.shape[0]) % int(num_samples)
    if remainder != 0:
        # This usually means the negative-sample CSV does not contain a fixed K negatives
        # per positive after filtering/subsetting. Truncate to the largest valid multiple.
        valid = (int(neg_scores.shape[0]) // int(num_samples)) * int(num_samples)
        logging.warning(
            "Neg/pos mismatch in evaluation (pos=%d, neg=%d, remainder=%d). "
            "Truncating negatives to %d to avoid reshape error. "
            "Recommended fix: regenerate val/test negative samples for the subset.",
            int(num_samples),
            int(neg_scores.shape[0]),
            int(remainder),
            int(valid),
        )
        if valid == 0:
            raise RuntimeError("No valid negatives available after truncation.")
        neg_scores = neg_scores[:valid]

    num_neg = int(neg_scores.shape[0]) // int(num_samples)
    neg_scores = neg_scores.view(int(num_samples), int(num_neg))
    
    candidates = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=-1)
    target = torch.zeros(num_samples, dtype=torch.long) # positive sample is always in 0 idx

    if dump_path is not None:
        # Per-positive dump for degree-stratified analysis: query user (src), partner (dst),
        # and the positive's mid-rank among its candidates. Order matches test_edge_index (shuffle=False).
        gt = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1).float()
        eq = (neg_scores == pos_scores.unsqueeze(1)).sum(dim=1).float()
        rank = (1.0 + gt + 0.5 * eq).cpu().numpy()
        np.savez(
            dump_path,
            src=pos_edge_index[0, :num_samples].cpu().numpy(),
            dst=pos_edge_index[1, :num_samples].cpu().numpy(),
            rank=rank,
            pos_score=pos_scores.detach().cpu().numpy(),
        )
        logging.info(f"[dump] {num_samples} per-edge test predictions -> {dump_path}")

    metrics = evaluate(candidates, target, k_list=hits_k)
    metrics['loss'] = loss.item()
    return metrics


def infer_bert_from_ckpt(ckpt_sd, default_layers=1, default_heads=2):
    """
    Infer vocab_size, hidden_size, num_hidden_layers, num_attention_heads from checkpoint.
    Falls back to defaults if not fully inferable.
    """
    # 1) vocab_size, hidden_size from embeddings
    emb_key = "sequence_encoder.bert.embeddings.word_embeddings.weight"
    if emb_key not in ckpt_sd:
        raise ValueError(f"Missing embedding key in ckpt: {emb_key}")
    vocab_size, hidden_size = ckpt_sd[emb_key].shape

    # 2) num_hidden_layers from encoder blocks present
    layer_indices = set()
    for k in ckpt_sd.keys():
        m = re.match(r"sequence_encoder\.bert\.encoder\.layer\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))
    num_hidden_layers = (max(layer_indices) + 1) if layer_indices else default_layers

    # 3) num_attention_heads is not stored directly; try to infer from filename or default.
    # We return heads=None here so the caller can override from filename; otherwise fallback.
    return vocab_size, hidden_size, num_hidden_layers, None  # heads unknown here

def infer_tokenxat_k_from_ckpt(ckpt_sd):
    """
    Infer compress_dim k from TokenXAT induced_queries shape [1, k, hidden].
    """
    for k in [
        "gnn.layers.0.compressor.induced_queries",  # primary
        "residual_compressor.induced_queries",      # backup
    ]:
        if k in ckpt_sd:
            shape = ckpt_sd[k].shape
            if len(shape) == 3 and shape[0] == 1:
                return int(shape[1])  # k
    return None


def infer_from_filename(fname):
    out = {}
    m = re.search(r"_bert(\d+)x(\d+)_", fname)
    if m:
        out["bert_layers"] = int(m.group(1))
        out["bert_heads"]  = int(m.group(2))
    m = re.search(r"_compress-dim(\d+)", fname)
    if m:
        out["compress_dim"] = int(m.group(1))
    return out

def build_model_matching_ckpt(args, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_sd = ckpt.get("model_state_dict", ckpt)

    # --- Infer from ckpt ---
    ck_vocab, ck_hidden, ck_layers, ck_heads_maybe = infer_bert_from_ckpt(ckpt_sd)
    k_from_ckpt = infer_tokenxat_k_from_ckpt(ckpt_sd)

    # --- Infer from filename (prefer filename for heads/layers if available) ---
    fname_hints = infer_from_filename(ckpt_path)
    bert_layers = fname_hints.get("bert_layers", ck_layers or args.bert_num_layers)
    bert_heads  = fname_hints.get("bert_heads", ck_heads_maybe or args.bert_num_heads)
    compress_dim = fname_hints.get("compress_dim", k_from_ckpt or args.compress_dim)

    # --- Override args to match ckpt shapes ---
    args.bert_hidden_size = ck_hidden
    args.bert_num_layers  = bert_layers
    args.bert_num_heads   = bert_heads
    args.compress_dim     = compress_dim

    # Build BertConfig with the *ckpt* vocab + hidden sizes
    max_seq_tokens = args.max_events * 4 + 1
    bert_config = BertConfig(
        vocab_size=ck_vocab,
        hidden_size=args.bert_hidden_size,
        num_hidden_layers=args.bert_num_layers,
        num_attention_heads=args.bert_num_heads,
        intermediate_size=args.bert_hidden_size * 2,
        max_position_embeddings=max_seq_tokens,
    )

    # Now instantiate your JointModel *using these overridden args*
    model = JointModel(
        bert_config=bert_config,
        gnn_in_channels=args.bert_hidden_size,
        gnn_hidden_channels=args.bert_hidden_size,
        gnn_num_layers=args.gnn_num_layers,
        gnn_dropout=args.gnn_dropout,
        gnn_heads=args.gnn_heads,
        args=args
    )
    return model, ckpt

def load_strict_or_filtered(model, ckpt):
    sd = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(sd, strict=True)
        print("[load] strict=True succeeded.")
    except RuntimeError as e:
        # breakpoint()
        print("[load] strict=True failed, falling back to exact-shape filter.", e)
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in sd.items()
            if (k in model_state) and (model_state[k].shape == v.shape)
        }
        missing = [k for k in model_state.keys() if k not in filtered]
        extra   = [k for k in sd.keys() if k not in model_state]
        mism    = [k for k in sd.keys() if (k in model_state) and (model_state[k].shape != sd[k].shape)]
        print(f"  loaded tensors: {len(filtered)}")
        print(f"  missing keys  : {len(missing)}")
        print(f"  extra keys    : {len(extra)}")
        print(f"  mismatched    : {len(mism)}")
        model.load_state_dict(filtered, strict=False)


# -----------------------------------------------------------------------------
# 4. Main Training and Evaluation Loop
# -----------------------------------------------------------------------------
def main(args):
    # ------------------- Setup -------------------
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # If user supplied a checkpoint to evaluate, switch to eval-only and return early
    EVAL_ONLY = args.load_checkpoint_and_eval is not None
    if EVAL_ONLY:
        ckpt_path = args.load_checkpoint_and_eval
        is_best_ckpt = "_best" in os.path.basename(ckpt_path)
        dump_predictions_requested = args.dump_predictions
        eval_batch_size_requested = args.eval_batch_size

        # Build results json path from checkpoint
        results_json_from_ckpt = (
            ckpt_path
            .replace('./','./')
            .replace("checkpoints", "results")
            .replace("_best.pth", ".json")
            .replace(".pth", ".json")
        )

        # Load args from checkpoint for consistent eval batch size, etc.
        ckpt = torch.load(ckpt_path, map_location="cpu")
        saved_args = ckpt.get("args", {})
        if isinstance(saved_args, dict):
            saved_args['load_checkpoint_and_eval'] = ckpt_path
            args = argparse.Namespace(**{**vars(args), **saved_args})
        # This eval-time request must not be overwritten by training-time args
        # stored in the checkpoint.
        args.dump_predictions = dump_predictions_requested
        args.eval_batch_size = (
            128 if 'tokenxat' in args.gnn_layer_types.lower()
            else eval_batch_size_requested
        )
        logger.info("Evaluation batch size: %d", args.eval_batch_size)
        set_seed(args.seed)

        # Data + model
        graph_data, uids, user_sequences_times = load_and_prepare_data(args)
        user_sequences = {uid: [e for e, t in seq] for uid, seq in user_sequences_times.items()}
        user_event_times = {uid: [t for e, t in seq] for uid, seq in user_sequences_times.items()}

        vocab = generate_vocab(args.csv_file, args.event_set)
        tokenizer = GeohashTokenizer(vocab); tokenizer.padding_side = 'left'
        max_seq_tokens = args.max_events * 4 + 1

        # Build per-node sequences (0..num_nodes-1)
        all_user_ids = range(graph_data.num_nodes)
        all_sequences_ids, all_sequences_timestamps = [], []
        for uid in all_user_ids:
            seq = user_sequences.get(uid, [])
            timestamp = user_event_times.get(uid, [])
            timestamp_by_4 = list(itertools.chain.from_iterable([[timestamp[i]]*4 for i in range(len(timestamp))])) + ([timestamp[-1]] if seq else [0])
            input_ids = tokenizer.encode(seq, max_length=max_seq_tokens, padding="max_length", truncation=True)
            pad_length = max_seq_tokens - len(timestamp) * 4 - 1
            pad_times = ([timestamp[-1]] if seq else [0]) * pad_length
            all_sequences_ids.append(input_ids)
            all_sequences_timestamps.append(pad_times + timestamp_by_4)

        user_sequences_ids = torch.tensor(all_sequences_ids, dtype=torch.long).to(device)
        user_sequences_masks = (user_sequences_ids != vocab["[PAD]"]).long().to(device)
        user_timestamps = torch.tensor(all_sequences_timestamps, dtype=torch.long).to(device)

        bert_config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=args.bert_hidden_size,
            num_hidden_layers=args.bert_num_layers,
            num_attention_heads=args.bert_num_heads,
            intermediate_size=args.bert_hidden_size * 2,
            max_position_embeddings=max_seq_tokens,
        )
        model, _ = build_model_matching_ckpt(args, ckpt_path)
        load_strict_or_filtered(model, ckpt)
        model.to(device)
        model.eval()

        # Criterion must match training
        criterion = F.binary_cross_entropy_with_logits

        # Evaluate on TEST
        test_metrics = test(
            model, model.predictor, graph_data,
            args.eval_batch_size, device, args.num_neighbors, args.hits_k,
            user_sequences_ids, user_sequences_masks, user_timestamps,
            criterion, split="test",
            dump_path=(os.path.splitext(ckpt_path)[0] + "_test_preds.npz") if getattr(args, "dump_predictions", False) else None
        )
        logger.info("Eval-only Test Metrics | " + " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))
        prefix = "best_test_" if is_best_ckpt else "test_"

        # Append a row into the results JSON
        prefixed = {f"{prefix}{k.lower().replace('@','_at_')}": float(v) for k, v in test_metrics.items()}
        prefixed.update({
            "eval_only": True,
            "checkpoint_used": os.path.basename(ckpt_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })
        append_row_to_json_log(results_json_from_ckpt, prefixed)
        logger.info(f"Appended eval-only metrics to {results_json_from_ckpt}")
        return

    # ------------------- Training path -------------------
    set_seed(args.seed)

    # Inputs & data
    args.input_path = args.input_path.replace('../data/tasks/','../data/tasks/')
    args.csv_file   = args.csv_file  .replace('../data/tasks/','../data/tasks/')

    graph_data, uids, user_sequences_times = load_and_prepare_data(args)
    user_sequences = {uid: [e for e, t in seq] for uid, seq in user_sequences_times.items()}
    user_event_times = {uid: [t for e, t in seq] for uid, seq in user_sequences_times.items()}

    vocab = generate_vocab(args.csv_file, args.event_set)
    tokenizer = GeohashTokenizer(vocab); tokenizer.padding_side = 'left'
    max_seq_tokens = args.max_events * 4 + 1

    results_dir = args.results_dir or os.path.join(args.output_root, f"{args.dataset}_results_seeded")
    checkpoints_dir = args.checkpoints_dir or os.path.join(args.output_root, f"{args.dataset}_checkpoints_seeded")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    gnn_layer_config_str = args.gnn_layer_types.lower().replace(',', '-')
    if "XAT" in args.gnn_layer_types.split(',')[0]:
        args.encoder_compress = 'stack'
        json_name = (
            f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}"
            f"{'_self-loop' if args.add_self_loops else ''}{'_qknorm' if args.qk_norm else ''}"
            f"{'_pre-agg-proj' if args.pre_agg_proj else ''}{'_post-agg-proj' if args.post_agg_proj else ''}"
            f"_{args.aggregation_mode}{'_compress-first' if args.compress_first else ''}"
            f"_compress-dim{args.compress_dim}{'_msg-mode-gcn' if args.msg_mode == 'gcn' else ''}"
            f"{'_layer-norm' if args.layer_norm else ''}{'_gcn-edge-weight' if args.gcn_edge_weight else ''}"
            f"{'_no-relational' if not args.include_relational else ''}"
            f"_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
            f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
            f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}.json"
            if not args.debug_subset else
            f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}"
            f"{'_self-loop' if args.add_self_loops else ''}{'_qknorm' if args.qk_norm else ''}"
            f"{'_pre-agg-proj' if args.pre_agg_proj else ''}{'_post-agg-proj' if args.post_agg_proj else ''}"
            f"_{args.aggregation_mode}{'_compress-first' if args.compress_first else ''}"
            f"_compress-dim{args.compress_dim}{'_msg-mode-gcn' if args.msg_mode == 'gcn' else ''}"
            f"{'_layer-norm' if args.layer_norm else ''}{'_gcn-edge-weight' if args.gcn_edge_weight else ''}"
            f"{'_no-relational' if not args.include_relational else ''}"
            f"_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
            f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
            f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_debug{args.debug_subset}_seed{args.seed}.json"
        )
    else:
        json_name = (
            f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}"
            f"{'_no-relational' if not args.include_relational else ''}"
            f"_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
            f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
            f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}.json"
            if not args.debug_subset else
            f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}"
            f"{'_no-relational' if not args.include_relational else ''}"
            f"_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
            f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
            f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_debug{args.debug_subset}_seed{args.seed}.json"
        )

    if getattr(args, 'run_tag', None):
        json_name = json_name[:-5] + f"_{args.run_tag}.json"

    results_save_path = os.path.join(results_dir, json_name)
    run_name = json_name.replace(".json","")

    if args.skip_if_exists and _results_json_has_test_metrics(results_save_path):
        logger.info(f"Results already exist (completed): {results_save_path}. Skipping.")
        return

    # Train loader
    train_loader = LinkNeighborLoader(
        graph_data,
        num_neighbors=[args.num_neighbors] * args.gnn_num_layers,
        edge_label_index=graph_data.train_edge_label_index,
        batch_size=args.batch_size,
        shuffle=True,
        neg_sampling_ratio=args.train_neg_sampling_ratio,
        num_workers=args.num_workers
    )

    # Model
    bert_config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=args.bert_hidden_size,
        num_hidden_layers=args.bert_num_layers,
        num_attention_heads=args.bert_num_heads,
        intermediate_size=args.bert_hidden_size * 2,
        max_position_embeddings=max_seq_tokens,
    )
    model = JointModel(
        bert_config=bert_config,
        gnn_in_channels=args.bert_hidden_size,
        gnn_hidden_channels=args.bert_hidden_size,
        gnn_num_layers=args.gnn_num_layers,
        gnn_dropout=args.gnn_dropout,
        gnn_heads=args.gnn_heads,
        args=args
    ).to(device)
    logger.info("--- Model Size ---")
    logger.info(f"Total Trainable Parameters: {count_parameters(model)} parameters")
    logger.info("------------------")

    # Build per-node sequences (0..num_nodes-1)
    all_user_ids = range(graph_data.num_nodes)
    all_sequences_ids, all_sequences_timestamps = [], []
    for uid in tqdm(all_user_ids, desc="Tokenizing sequences"):
        seq = user_sequences.get(uid, [])
        timestamp = user_event_times.get(uid, [])
        timestamp_by_4 = list(itertools.chain.from_iterable([[timestamp[i]]*4 for i in range(len(timestamp))])) + ([timestamp[-1]] if seq else [0])
        input_ids = tokenizer.encode(seq, max_length=max_seq_tokens, padding="max_length", truncation=True)
        pad_length = max_seq_tokens - len(timestamp)*4 - 1
        pad_times = ([timestamp[-1]] if seq else [0]) * pad_length
        all_sequences_ids.append(input_ids)
        all_sequences_timestamps.append(pad_times + timestamp_by_4)

    user_sequences_ids = torch.tensor(all_sequences_ids, dtype=torch.long).to(device)
    user_sequences_masks = (user_sequences_ids != vocab["[PAD]"]).long().to(device)
    user_timestamps = torch.tensor(all_sequences_timestamps, dtype=torch.long).to(device)

    # Fair controls: load a masked-LM-pretrained sequence encoder.
    #   --pretrained_seq_ckpt alone   => unfrozen two-stage (pretrained encoder, fine-tuned jointly)
    #   --pretrained_seq_ckpt + freeze => architecture-matched frozen two-stage
    # ckpt is an MLMModel state_dict (sequence_embedding.py); we transfer only its bert.* weights.
    if getattr(args, 'pretrained_seq_ckpt', None):
        sd = torch.load(args.pretrained_seq_ckpt, map_location=device)
        sd = sd.get('model_state_dict', sd)
        bert_sd = {k[len('bert.'):]: v for k, v in sd.items() if k.startswith('bert.')}
        msg = model.sequence_encoder.bert.load_state_dict(bert_sd, strict=False)
        logger.info(f"[pretrained-seq] loaded {len(bert_sd)} bert tensors from "
                    f"{args.pretrained_seq_ckpt} (missing={len(msg.missing_keys)}, "
                    f"unexpected={len(msg.unexpected_keys)})")
        if args.freeze_seq_encoder:
            for p in model.sequence_encoder.parameters():
                p.requires_grad = False
            logger.info("[pretrained-seq] sequence encoder FROZEN (frozen two-stage control)")

    # Supervised-frozen control (variant D): task-supervised sequence-only warm-up -> freeze -> GNN.
    # Distinct from the (i) frozen-MLM control above -- the encoder is trained on the LINK target with
    # NO graph message passing, not loaded from the unsupervised MLM ckpt. Do not combine with
    # --freeze_seq_encoder. The optimizer below (filter requires_grad) then trains the GNN only.
    if getattr(args, 'supervised_warmup_epochs', 0) > 0 and not args.freeze_seq_encoder:
        logging.info(f"[supervised-frozen] Phase A: warming up the sequence encoder on the link task "
                     f"for {args.supervised_warmup_epochs} epochs (sequence-only, no graph)...")
        supervised_seq_warmup(model, train_loader, user_sequences_ids, user_sequences_masks,
                              device, args.supervised_warmup_epochs, args.lr)
        for p in model.sequence_encoder.parameters():
            p.requires_grad = False
        logging.info("[supervised-frozen] Phase B: sequence encoder FROZEN after supervised warm-up "
                     "(variant D); training the graph module only.")

    # Loss/opt
    criterion = F.binary_cross_entropy_with_logits
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # ---------- Initial metrics ----------
    initial_metrics_log = []
    args.num_negative_samples = graph_data.val_neg_edge_index.shape[1] // graph_data.val_edge_index.shape[1]
    num_batches = len(train_loader)
    model.eval()
    total_initial_loss = 0.0
    t0 = time.time()
    average_initial_loss = total_initial_loss / max(1, num_batches)
    epoch_time = time.time() - t0
    logging.info(f"Initial Average Training Loss (Full Dataset): {average_initial_loss:.4f}")
    gpu_mem_allocated = gpu_mem_peak = 0.0
    if device.type == 'cuda':
        gpu_mem_allocated = round(torch.cuda.memory_allocated(device) / (1024**2), 2)
        gpu_mem_peak = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        torch.cuda.reset_peak_memory_stats(device)

    eval_begin = time.time()
    initial_val_metrics = test(
        model, model.predictor, graph_data,
        args.eval_batch_size, device, args.num_neighbors, args.hits_k,
        user_sequences_ids, user_sequences_masks, user_timestamps,
        criterion, split="val"
    )
    eval_time = time.time() - eval_begin
    val_mrr = initial_val_metrics.get("MRR", 0.0)
    val_link_loss = initial_val_metrics.get("loss", 0.0)
    log_msg_val = f"Initial Pre-Training Metrics (Val) | Loss: {val_link_loss:.4f}, MRR: {val_mrr:.4f}"
    for k in args.hits_k:
        log_msg_val += f", Hits@{k}: {initial_val_metrics.get(f'Hits@{k}', 0.0):.4f}"
    logging.info(log_msg_val)

    epoch0_row = {
        'epoch': 0,
        'train_loss': average_initial_loss,
        "evaluation_metrics": initial_val_metrics,
        "epoch_time": epoch_time,
        "eval_time": eval_time,
        "gpu_mem_allocated_mb": gpu_mem_allocated,
        "gpu_mem_peak_mb": gpu_mem_peak
    }
    initial_metrics_log.append(epoch0_row)

    # ---------- Checkpoint manager ----------
    checkpoint_path = os.path.join(checkpoints_dir, os.path.basename(results_save_path).replace(".json", ".pth"))
    best_manager = BestCheckpointManager(
        base_checkpoint_path=checkpoint_path,
        mode="max"  # monitoring Val MRR
    )

    # ---------- Optional resume from last checkpoint (model + optimizer + epoch) ----------
    start_epoch = 1
    if getattr(args, "resume", False) and os.path.exists(checkpoint_path):
        rckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(rckpt["model_state_dict"])
        optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        start_epoch = int(rckpt.get("epoch", 0)) + 1
        logger.info(f"[resume] {checkpoint_path} @ epoch {rckpt.get('epoch')} -> resuming at epoch {start_epoch} "
                    f"(early-stopping/RNG reset; fixed-LR Adam state restored)")
    elif getattr(args, "resume", False):
        logger.info(f"[resume] no checkpoint at {checkpoint_path}; starting fresh")

    # ---------- Train ----------
    training_results, best_model_state, best_metric, best_time = train_model(
        model,
        train_loader,
        graph_data,
        user_sequences_ids,
        user_sequences_masks,
        user_timestamps,
        optimizer,
        tokenizer,
        device,
        criterion,
        args,
        initial_metrics_log=initial_metrics_log,
        log_path=results_save_path,
        run_name=run_name,
        ckpt_manager=best_manager,
        start_epoch=start_epoch,
    )

    # ---------- Final and Best test ----------
    # Evaluate final
    final_test = test(
        model, model.predictor, graph_data,
        args.eval_batch_size, device, args.num_neighbors, args.hits_k,
        user_sequences_ids, user_sequences_masks, user_timestamps,
        criterion, split="test"
    )
    final_test_pref = prefix_metrics(final_test, "test_")
    logging.info("Final Test Metrics | " + " | ".join(f"{k}: {v:.4f}" for k, v in final_test_pref.items()))

    # Save last checkpoint with final metrics
    save_checkpoint(
        filepath=checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=args,
        elapsed_time=sum(ep.get('epoch_time', 0) for ep in training_results),
        test_metrics=final_test_pref
    )

    # Optionally evaluate & save best
    best_test_pref = None
    if best_model_state is not None:
        best_checkpoint_path = checkpoint_path.replace('.pth', '_best.pth')
        model.load_state_dict(best_model_state)
        model.eval()
        best_test = test(
            model, model.predictor, graph_data,
            args.eval_batch_size, device, args.num_neighbors, args.hits_k,
            user_sequences_ids, user_sequences_masks, user_timestamps,
            criterion, split="test",
            dump_path=(os.path.splitext(best_checkpoint_path)[0] + "_test_preds.npz") if getattr(args, "dump_predictions", False) else None
        )
        best_test_pref = prefix_metrics(best_test, "best_test_")
        logging.info("Best-Checkpoint Test Metrics | " + " | ".join(f"{k}: {v:.4f}" for k, v in best_test_pref.items()))
        save_checkpoint(
            filepath=best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            args=args,
            elapsed_time=sum(ep.get('epoch_time', 0) for ep in training_results),
            test_metrics=best_test_pref
        )

    # ---------- Persist JSON log ----------
    # Append terminal rows for visibility
    training_results.append(final_test_pref)
    if best_test_pref is not None:
        training_results.append(best_test_pref)
    with open(results_save_path.replace('.csv', '.json'), 'w') as f:
        json.dump(training_results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Joint Training of Sequence and Graph Models")
    
    # Data Args
    parser.add_argument("--dataset", type=str, choices=['brightkite','gowalla'], default='brightkite')
    parser.add_argument("--input_path", type=str, default=None,
                        help="Directory containing relational CSVs (overrides dataset default if set).")
    parser.add_argument("--csv_file", type=str, default=None,
                        help="Path to personal.csv (defaults to <input_path>/personal.csv if not set).")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root folder for writing result logs and checkpoints.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to write JSON logs (overrides output_root default).")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Directory to write checkpoints (overrides output_root default).")
    parser.add_argument("--skip_if_exists", action="store_true",
                        help="Skip training/eval if the result JSON already exists and contains test metrics.")
    parser.add_argument("--event_set", type=str, default="personal")
    parser.add_argument('--eval_every', type=int, default=10,
                    help="Evaluation frequency (in epochs).")
    parser.add_argument("--debug_subset", type=int, default=None, help="Run on a small subset of N training edges for debugging.")
    parser.add_argument("--max_len", type=int, default=100)

    
    # Sequence Model (BERT) Args
    parser.add_argument("--max_events", type=int, default=50)
    parser.add_argument("--max_examples", type=int, default=50, help="Max number of examples to generate per user")
    parser.add_argument("--bert_hidden_size", type=int, default=128)
    parser.add_argument("--bert_num_layers", type=int, default=1)
    parser.add_argument("--bert_num_heads", type=int, default=2)
    parser.add_argument("--num_negative_samples", type=int, default=100, help="Number of negative candidates per sample")
    
    # Graph Model (GNN) Args
    parser.add_argument('--gnn_model', type=str, choices=['GCN', 'GraphSAGE', 'GAT', 'TokenXAT', 'TokenXLinAT','GATv2','TransformerConv'], default='GCN')
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--gnn_heads', type=int, default=1)
    parser.add_argument('--gnn_layer_types', type=str, default='TokenXAT,GCN')
    parser.add_argument('--run_diagnostic_gcn', action='store_true',
                    help='If set, runs the simplified GCNWithTokenAggregation baseline.')
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--add_self_loops', action='store_true')
    parser.add_argument('--aggregation_mode', type=str, default='fixed')
    parser.add_argument('--compress_dim', type=int, default=4)
    parser.add_argument('--qk_norm', action='store_true')
    parser.add_argument('--compress_first', action='store_true')
    parser.add_argument('--msg_mode',type=str, choices=['gat','gcn'], default='gat',help="Test if TokenCrossAttn is equivalent to gat/ gcn when compressed first")
    parser.add_argument('--gcn_edge_weight', action='store_true')
    # Training Args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=128 * 2,
                    help="Mini-batch size for evaluation and testing.")
    parser.add_argument('--patience', type=int, default=20,
                        help="Early stopping patience (in epochs).")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help="Maximum norm for gradient clipping.")
    parser.add_argument('--train_neg_sampling_ratio', type=float, default=2.0)
    parser.add_argument('--loss_alpha', type=float, default=0.5, help="Weight for link prediction loss (1-alpha for MLM loss)")
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hits_k', type=str, default="1,3,5,10,20,50,100",
                        help="Comma separated list of k values for computing Hits@k metrics.")
    parser.add_argument('--enable_early_stopping', action='store_true',
                        help='Enable early stopping based on validation MRR. (default: False)')
    # New argument for experiment name (for saving artifacts)
    parser.add_argument('--experiment_name', type=str, default='PG-exp-1',
                        help="Experiment name for saving checkpoints and artifacts.")
    # NEW: Option to adjust number of workers used in the data loaders.
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of worker processes for data loaders (default: 0 (all))")
    # NEW: Option to include or exclude relational edges.
    parser.add_argument('--no_relational', dest='include_relational', action='store_false',
                        help="Exclude relational edges from the graph.")
    parser.add_argument('--encoder_compress', type=str, choices=['cls', 'mean','stack','4d'], default='mean')
    parser.add_argument('--pre_agg_proj', action='store_true')
    parser.add_argument('--post_agg_proj', action='store_true')
    parser.set_defaults(include_relational=True)
    parser.add_argument('--use_wandb', action='store_true',
                help="Enable logging to Weights & Biases.")
    parser.add_argument('--layer_norm', action='store_true',
                help="Use layer_norm.")
    parser.add_argument('--log_interval', type=int, default=100, help="Log batch loss every N steps.")
    # --- ADD THIS ARGUMENT ---
    parser.add_argument(
        '--load_checkpoint_and_eval', 
        type=str, 
        # default="./checkpoints_seeded/e2e_gcn-gcn2x1_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs128_lr0.0001_epochs100_seed2.pth",
        default=None,
        help="Path to a model checkpoint to load for evaluation only. Skips training."
    )
    parser.add_argument('--pretrained_seq_ckpt', type=str, default=None,
                        help="Path to a masked-LM-pretrained sequence-encoder checkpoint "
                             "(MLMModel state_dict from sequence_embedding.py). Transfers bert.* "
                             "weights into the sequence encoder for the unfrozen two-stage control.")
    parser.add_argument('--freeze_seq_encoder', action='store_true',
                        help="Freeze the sequence encoder after loading pretrained weights "
                             "(architecture-matched frozen two-stage control).")
    parser.add_argument('--supervised_warmup_epochs', type=int, default=0,
                        help="Supervised-frozen control (variant D): epochs to train the sequence "
                             "encoder on the LINK target sequence-only (no graph) before freezing it "
                             "and training the GNN. 0 = off. Do NOT combine with --freeze_seq_encoder "
                             "(that is the unsupervised MLM-frozen control (i)).")
    parser.add_argument('--run_tag', type=str, default=None,
                        help="Suffix appended to the result/checkpoint filename so control "
                             "variants (unfrozen/frozen/bridge) don't collide or falsely skip.")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from this run's last saved checkpoint "
                             "(restores model + optimizer + epoch; ~eval_every-epoch granularity). "
                             "Pairs with --skip_if_exists: completed runs skip, partial runs resume.")
    parser.add_argument('--dump_predictions', action='store_true',
                        help="In eval-only mode (--load_checkpoint_and_eval), dump per-edge test "
                             "predictions (src, dst, rank) to <ckpt>_test_preds.npz for the "
                             "degree-stratified analysis (degree_stratified.py).")
    parser.add_argument('--args_from_checkpoint', type=str, default=None,
                        help="Inherit the training configuration saved in this checkpoint's 'args'. "
                             "Flags given explicitly on the command line override it; machine- and "
                             "run-specific keys (paths, GPU, output/resume controls) are never inherited.")
    parser.add_argument('--print_args', action='store_true',
                        help="Print the effective arguments (and, with --args_from_checkpoint, every "
                             "difference from the checkpoint's saved args), then exit without training.")

    args = parser.parse_args()

    ref_args = None
    if args.args_from_checkpoint:
        ref_args = torch.load(args.args_from_checkpoint, map_location="cpu", weights_only=False).get("args")
        if not isinstance(ref_args, dict):
            raise SystemExit(f"No saved 'args' dict in {args.args_from_checkpoint}")
        # dests the user set explicitly on the command line
        option_to_dest = {opt: a.dest for a in parser._actions for opt in a.option_strings}
        explicit = {option_to_dest[tok.split('=', 1)[0]] for tok in sys.argv[1:]
                    if tok.split('=', 1)[0] in option_to_dest}
        not_inherited = {
            'output_root', 'input_path', 'csv_file', 'results_dir', 'checkpoints_dir',
            'gpu_id', 'num_workers', 'use_wandb', 'experiment_name', 'run_tag',
            'skip_if_exists', 'resume', 'resume_if_possible', 'load_checkpoint_and_eval',
            'dump_predictions', 'args_from_checkpoint', 'print_args',
        }
        known = set(vars(args))
        for k, v in ref_args.items():
            if k in known and k not in explicit and k not in not_inherited:
                setattr(args, k, v)
        unknown = sorted(set(ref_args) - known)
        if unknown:
            logger.warning("Checkpoint args not recognized by this joint.py (ignored): %s", unknown)

    if isinstance(args.hits_k, str):
        args.hits_k = [int(k) for k in args.hits_k.split(',')]

    if args.output_root is None:
        args.output_root = os.path.abspath(os.path.dirname(__file__))
    
    if args.input_path is None:
        if args.dataset == "brightkite":
            args.input_path = "../data/tasks/brightkite/friend_recommendation"
        elif args.dataset == "gowalla":
            args.input_path = "../data/tasks/gowalla/friend_recommendation"

    if args.csv_file is None:
        args.csv_file = os.path.join(args.input_path, "personal.csv")

    if args.print_args:
        for k in sorted(vars(args)):
            print(f"{k} = {getattr(args, k)!r}")
        if ref_args is not None:
            print(f"\n# differences from {args.args_from_checkpoint}")
            for k in sorted(set(vars(args)) | set(ref_args)):
                new, old = getattr(args, k, '<absent>'), ref_args.get(k, '<absent>')
                if new != old:
                    print(f"{k}: {old!r} -> {new!r}")
        sys.exit(0)

    main(args)
