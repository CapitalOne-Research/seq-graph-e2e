import argparse
import logging
import pickle
import random
import json
import os
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
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
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
from collections import OrderedDict
from torch.nn import Parameter, ModuleList, MultiheadAttention
import wandb
import copy
import re
# # Better: combine with an environment variable
# api_key = os.getenv("WANDB_API_KEY")
# if api_key:
#     wandb.login(key=api_key)



# -----------------------------------------------------------------------------
# Configure Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            elif model_type == 'GAT':
                # GAT's output is concatenated, so it becomes hidden_channels
                layer = GATConv(current_in_channels,  current_in_channels// heads, heads=heads, dropout=dropout, residual = True)    
            elif model_type == 'GATv2':
                # GAT's output is concatenated, so it becomes hidden_channels
                layer = GATv2Conv(current_in_channels,  current_in_channels// heads, heads=heads, dropout=dropout, residual = True)
            elif model_type == 'TransformerConv':
                # GAT's output is concatenated, so it becomes hidden_channels
                layer = TransformerConv(current_in_channels,  current_in_channels// heads, heads=heads, dropout=dropout)
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
        # breakpoint()
        # Select the sequences and masks for the users in the current computational graph
        batch_seq_ids = user_sequences_ids[all_user_ids_in_batch]
        batch_seq_masks = user_sequences_masks[all_user_ids_in_batch]
        user_embeddings = self.sequence_encoder(batch_seq_ids, batch_seq_masks)

        node_features_after_gnn = self.gnn(user_embeddings, graph_data.edge_index, pad_mask = batch_seq_masks,training = training)
        

        if user_embeddings.dim() == 3:
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
        # breakpoint()
        combined_features = residual_features + node_features_after_gnn
        

        return combined_features


def load_and_prepare_data(args):
    """
    Loads and preprocesses all data consistently, returning model-ready objects
    and a mapping to the original IDs.
    """
    logger.info("Starting data loading and preparation...")
    
    # --- 1. Load all raw dataframes ---
    train_df = pd.read_csv(os.path.join(args.input_path, "relational_train.csv"), nrows=args.debug_subset)
    val_df = pd.read_csv(os.path.join(args.input_path, "relational_val.csv"))
    if args.num_negative_samples < 1000:
        val_neg_df = pd.read_csv(os.path.join(args.input_path, "relational_val_negative_sample_100.csv"))
    else:
        val_neg_df = pd.read_csv(os.path.join(args.input_path, "relational_val_negative_sample.csv"))
    seq_df = pd.read_csv(args.csv_file)

    # --- 2. Handle Debug Subsetting (Optional) ---
    if args.debug_subset and args.debug_subset > 0:
        logger.info(f"--- Running in DEBUG mode: creating a small, self-contained graph ---")
        active_nodes_subset = pd.unique(train_df[['uid', 'other_uid']].values.ravel('K'))
        val_df = val_df[val_df['uid'].isin(active_nodes_subset) & val_df['other_uid'].isin(active_nodes_subset)]
        val_neg_df = val_neg_df[val_neg_df['uid'].isin(active_nodes_subset) & val_neg_df['other_uid'].isin(active_nodes_subset)]
        seq_df = seq_df[seq_df['uid'].isin(active_nodes_subset)]

    # --- 3. Establish Single Source of Truth for ALL Nodes ---
    all_train_nodes = pd.unique(train_df[['uid', 'other_uid']].values.ravel('K'))
    all_val_nodes = pd.unique(val_df[['uid', 'other_uid']].values.ravel('K'))
    all_val_neg_nodes = pd.unique(val_neg_df[['uid', 'other_uid']].values.ravel('K'))
    all_seq_nodes = pd.unique(seq_df['uid'])
    
    all_nodes_combined = np.concatenate([all_train_nodes, all_val_nodes, all_val_neg_nodes, all_seq_nodes])
    all_unique_nodes = sorted([node for node in pd.unique(all_nodes_combined) if pd.notna(node)])
    
    node_map = {old_id: new_id for new_id, old_id in enumerate(all_unique_nodes)}
    num_nodes = len(node_map)
    logger.info(f"Found {num_nodes} unique nodes across all data sources.")

    # --- 4. Apply the single, consistent mapping to all dataframes ---
    def remap_df_safe(df, node_map, columns):
        for col in columns:
            df[col] = df[col].map(node_map)
        return df.dropna(subset=columns).astype({col: int for col in columns})

    train_df = remap_df_safe(train_df, node_map, ['uid', 'other_uid'])
    val_df = remap_df_safe(val_df, node_map, ['uid', 'other_uid'])
    val_neg_df = remap_df_safe(val_neg_df, node_map, ['uid', 'other_uid'])
    seq_df = remap_df_safe(seq_df, node_map, ['uid'])
    
    # --- 5. Create Graph Object from remapped data ---
    data = Data(num_nodes=num_nodes)
    data.edge_index = torch.tensor(train_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    data = T.ToUndirected()(data)
    data.val_edge_index = torch.tensor(val_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    data.val_neg_edge_index = torch.tensor(val_neg_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    logger.info(f"Graph loaded: {data}")
    # --- 5.5 Load test edges (Amazon tasks provide these files) ---
    test_path = args.input_path  # e.g., ../data/tasks/amazon-movies/coreview_prediction/
    test_df = pd.read_csv(os.path.join(test_path, "relational_test.csv"))
    # Prefer the smaller 100-neg version if you use it elsewhere consistently:
    test_neg_csv = "relational_test_negative_sample_100.csv"
    if not os.path.exists(os.path.join(test_path, test_neg_csv)):
        test_neg_csv = "relational_test_negative_sample.csv"
    test_neg_df = pd.read_csv(os.path.join(test_path, test_neg_csv))
    # breakpoint()

    # Remap test splits using the same global node_map
    test_df = remap_df_safe(test_df, node_map, ['uid', 'other_uid'])
    test_neg_df = remap_df_safe(test_neg_df, node_map, ['uid', 'other_uid'])

    data.test_edge_index = torch.tensor(test_df[['uid', 'other_uid']].values.T, dtype=torch.long)
    data.test_neg_edge_index = torch.tensor(test_neg_df[['uid', 'other_uid']].values.T, dtype=torch.long)

    # --- 6. Process Sequences using remapped IDs ---
    seq_df = seq_df.sort_values(by=["uid", "timestamp"])
    user_groups = seq_df.groupby("uid")[['event', 'timestamp']].apply(lambda g: list(zip(g['event'], g['timestamp'])))
    partial_sequences = {uid: seq[-args.max_events:] for uid, seq in user_groups.items()}
    user_sequences = {i: partial_sequences.get(i, []) for i in range(num_nodes)}
    
    logger.info(f"Loaded sequences for {len(partial_sequences)} users, created empty sequences for {num_nodes - len(partial_sequences)} users.")
    
    # --- 7. Return the final, consistent objects ---
    return data, user_sequences, node_map


# def load_and_prepare_data(args):
#     """Loads and preprocesses all data consistently for both normal and debug modes."""
#     logger.info("Starting data loading and preparation...")
    
#     # --- 1. Load all raw dataframes ---
#     # Use nrows to limit the initial training set size in debug mode
#     train_df = pd.read_csv(os.path.join(args.input_path, "relational_train.csv"), nrows=args.debug_subset)
#     val_df = pd.read_csv(os.path.join(args.input_path, "relational_val.csv"))
#     if args.num_negative_samples < 1000:
#         val_neg_df = pd.read_csv(os.path.join(args.input_path, "relational_val_negative_sample_100.csv"))
#     else:
#         val_neg_df = pd.read_csv(os.path.join(args.input_path, "relational_val_negative_sample.csv"))
#     seq_df = pd.read_csv(args.csv_file)

#     # --- 2. Handle Debug Subsetting (Optional) ---
#     if args.debug_subset and args.debug_subset > 0:
#         logger.info(f"--- Running in DEBUG mode: creating a small, self-contained graph ---")
#         # Identify a small set of root nodes from the initial training subset
#         active_nodes_subset = pd.unique(train_df[['uid', 'other_uid']].values.ravel('K'))
#         # Filter all other dataframes to only include interactions involving these nodes
#         val_df = val_df[val_df['uid'].isin(active_nodes_subset) & val_df['other_uid'].isin(active_nodes_subset)]
#         val_neg_df = val_neg_df[val_neg_df['uid'].isin(active_nodes_subset) & val_neg_df['other_uid'].isin(active_nodes_subset)]
#         seq_df = seq_df[seq_df['uid'].isin(active_nodes_subset)]

#     # --- 3. Establish Single Source of Truth for ALL Nodes ---
#     # Find all unique node IDs across ALL relevant dataframes (whether subsetted or full)
#     all_train_nodes = pd.unique(train_df[['uid', 'other_uid']].values.ravel('K'))
#     all_val_nodes = pd.unique(val_df[['uid', 'other_uid']].values.ravel('K'))
#     all_val_neg_nodes = pd.unique(val_neg_df[['uid', 'other_uid']].values.ravel('K'))
#     all_seq_nodes = pd.unique(seq_df['uid'])
    
#     all_nodes_combined = np.concatenate([all_train_nodes, all_val_nodes, all_val_neg_nodes, all_seq_nodes])
#     all_unique_nodes = sorted([node for node in pd.unique(all_nodes_combined) if pd.notna(node)])
#     breakpoint()
    
#     # Create one definitive mapping from original ID to new ID (0 to N-1)
#     node_map = {old_id: new_id for new_id, old_id in enumerate(all_unique_nodes)}
#     num_nodes = len(node_map)
#     logger.info(f"Found a total of {num_nodes} unique nodes across all data sources.")

#     # --- 4. Apply the single, consistent mapping to all dataframes ---
#     def remap_df(df, node_map, columns_to_remap):
#         for col in columns_to_remap:
#             if col in df.columns:
#                 df[col] = df[col].map(node_map)
#         return df.dropna().astype({col: int for col in columns_to_remap if col in df.columns})

#     train_df = remap_df(train_df, node_map, ['uid', 'other_uid'])
#     val_df = remap_df(val_df, node_map, ['uid', 'other_uid'])
#     val_neg_df = remap_df(val_neg_df, node_map, ['uid', 'other_uid'])
#     seq_df = remap_df(seq_df, node_map, ['uid'])
    
#     # --- 5. Create Graph and Sequence Objects from Clean Data ---
#     edge_index = torch.tensor(train_df[['uid', 'other_uid']].values.T, dtype=torch.long)
#     data = Data(edge_index=edge_index, num_nodes=num_nodes)
#     data = T.ToUndirected()(data)
    
#     data.val_edge_index = torch.tensor(val_df[['uid', 'other_uid']].values.T, dtype=torch.long)
#     data.val_neg_edge_index = torch.tensor(val_neg_df[['uid', 'other_uid']].values.T, dtype=torch.long)
#     logger.info(f"Graph loaded: {data}")

#     # Process sequences (uids are now the new, mapped IDs)
#     if args.event_set:
#         seq_df = seq_df[seq_df["event_set"] == args.event_set]
#     seq_df = seq_df.sort_values(by=["uid", "timestamp"])
    
#     user_groups = seq_df.groupby("uid")[['event', 'timestamp']].apply(
#         lambda g: list(zip(g['event'], g['timestamp']))
#     )
#     user_sequences = {uid: seq[-args.max_events:] for uid, seq in user_groups.items()}
#     uids = sorted(list(user_sequences.keys()))
    
#     logger.info(f"Loaded sequences for {len(user_sequences)} users.")
#     return data, uids, user_sequences

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
    model_state = model.state_dict()
    
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



def _strip_module_prefix(state_dict):
    """Strip a leading 'module.' (from DDP saves) if present."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}

def _maybe_rename_keys(state_dict, rename_map=None):
    """
    Optionally rename keys by prefix. Example:
        rename_map = {'sequence_encoder2.': 'sequence_encoder.', 'gnn2.': 'gnn.'}
    Only the first matching prefix is applied per key.
    """
    if not rename_map:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in rename_map.items():
            if new_k.startswith(old):
                new_k = new + new_k[len(old):]
                break
        out[new_k] = v
    return out

def safe_load_checkpoint(
    model,
    ckpt_path,
    device="cpu",
    rename_map=None,
    min_loaded_frac=0.0,   # e.g., 0.8 to require at least 80% of params matched
    verbose=True,
):
    """
    Load a checkpoint *safely*:
      - only load intersection keys with identical shapes,
      - silently ignore keys not present in the model,
      - report skipped due to shape mismatch,
      - optionally rename key prefixes (refactors),
      - optionally enforce a minimum fraction of matched params.

    Returns:
      (ckpt, report) where `ckpt` is the loaded checkpoint object (dict),
      and `report` contains coverage and lists of skipped/missing/unexpected keys.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)  # support wrapped or raw
    state_dict = _strip_module_prefix(state_dict)
    state_dict = _maybe_rename_keys(state_dict, rename_map=rename_map)

    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    unexpected_ckpt = []  # keys in ckpt not found in model

    # Keep only keys present in the model with identical shapes
    for k, w in state_dict.items():
        if k in model_state:
            if model_state[k].shape == w.shape:
                filtered[k] = w
            else:
                skipped_shape.append((k, tuple(w.shape), tuple(model_state[k].shape)))
        else:
            unexpected_ckpt.append(k)

    # Actually load the filtered subset
    msg = model.load_state_dict(filtered, strict=False)

    # Build a small report
    total_params = len(model_state)
    loaded_params = len(filtered)
    loaded_frac = loaded_params / max(total_params, 1)

    missing_in_ckpt = list(msg.missing_keys)        # present in model, not provided by ckpt
    unexpected_in_load = list(msg.unexpected_keys)  # PyTorch's view of unexpected (should be rare here)

    report = {
        "loaded_count": loaded_params,
        "total_model_params": total_params,
        "loaded_frac": loaded_frac,
        "skipped_shape_mismatch": skipped_shape,
        "missing_in_ckpt": missing_in_ckpt,
        "unexpected_in_ckpt": sorted(set(unexpected_ckpt + unexpected_in_load)),
    }

    if verbose:
        print(f"[safe_load] Loaded {loaded_params}/{total_params} tensors ({loaded_frac:.1%}).")
        if skipped_shape:
            print(f"[safe_load] Skipped {len(skipped_shape)} due to shape mismatch (showing up to 10):")
            for name, s_ckpt, s_model in skipped_shape[:10]:
                print(f"  - {name}: ckpt={s_ckpt}, model={s_model}")
            if len(skipped_shape) > 10:
                print(f"  ... and {len(skipped_shape)-10} more")
        if missing_in_ckpt:
            print(f"[safe_load] Missing in ckpt (exist in model, not loaded): {len(missing_in_ckpt)}")
            for k in missing_in_ckpt[:10]:
                print("  -", k)
            if len(missing_in_ckpt) > 10:
                print(f"  ... and {len(missing_in_ckpt)-10} more")
        if report["unexpected_in_ckpt"]:
            print(f"[safe_load] Unexpected keys from ckpt (ignored): {len(report['unexpected_in_ckpt'])}")
            for k in report["unexpected_in_ckpt"][:10]:
                print("  -", k)
            if len(report["unexpected_in_ckpt"]) > 10:
                print(f"  ... and {len(report['unexpected_in_ckpt'])-10} more")

    if loaded_frac < min_loaded_frac:
        raise RuntimeError(
            f"[safe_load] Only {loaded_frac:.1%} of model tensors matched the checkpoint. "
            f"This likely indicates an architecture/tokenizer mismatch."
        )

    return ckpt, report



# def safe_load_checkpoint(model, ckpt_path, device):
#     """
#     Load only the keys from the checkpoint that exist in the current model
#     and have matching shapes.
#     """
#     ckpt = torch.load(ckpt_path, map_location=device)
#     state_dict = ckpt.get("model_state_dict", ckpt)  # handle wrapped dicts
    
#     model_state = model.state_dict()
#     filtered_state = {}
#     skipped = []
    
#     for k, v in state_dict.items():
#         if k in model_state:
#             if model_state[k].shape == v.shape:
#                 filtered_state[k] = v
#             else:
#                 skipped.append((k, v.shape, model_state[k].shape))
    
#     # Actually load
#     msg = model.load_state_dict(filtered_state, strict=False)
    
#     print(f"[safe_load] Loaded {len(filtered_state)} matching tensors.")
#     if skipped:
#         print(f"[safe_load] Skipped {len(skipped)} due to shape mismatch:")
#         for name, ckpt_shape, model_shape in skipped[:10]:
#             print(f"  - {name}: ckpt={ckpt_shape}, model={model_shape}")
#         if len(skipped) > 10:
#             print(f"  ... and {len(skipped)-10} more")
    
#     if msg.missing_keys:
#         print(f"[safe_load] Missing keys in checkpoint: {len(msg.missing_keys)}")
#         for k in msg.missing_keys[:10]:
#             print("  -", k)
#     if msg.unexpected_keys:
#         print(f"[safe_load] Unexpected keys in checkpoint: {len(msg.unexpected_keys)}")
#         for k in msg.unexpected_keys[:10]:
#             print("  -", k)
    
#     return ckpt

logger = logging.getLogger(__name__)


# Assume a logger is configured, e.g., logger = logging.getLogger(__name__)

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
    run_name = None):
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

    for epoch in range(1, args.num_epochs + 1):
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
            if args.use_wandb:
                running_interval_loss += loss_item
            
            pbar.set_postfix({'TrainLink': f'{total_link_loss / (i+1):.4f}'})

            
            # --- Log Batch-level Metrics at Intervals ---
            if args.use_wandb and global_step % args.log_interval == 0:
                wandb.log({
                    'train/batch_loss': running_interval_loss / args.log_interval
                }, step=global_step)
                running_interval_loss = 0.0 # Reset for next interval

        
                
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
                print(f"New best model found at epoch {epoch+1}")

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
            
        # --- Log Epoch Summary to W&B ---
        if args.use_wandb:
            epoch_log_data = {
                'epoch': epoch,
                'train/epoch_loss': avg_epoch_train_loss,
            }
            if metrics: # Add validation metrics if they were computed
                for key, value in metrics.items():
                    epoch_log_data[f'val/{key}'] = value
            
            wandb.log(epoch_log_data, step=global_step)
            
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

def test(model, predictor, data, eval_batch_size, device, num_neighbors, hits_k, user_sequences_ids, user_sequences_masks, user_timestamps, criterion, split="test"):
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
            # breakpoint()
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
    num_neg = neg_scores.shape[0] // num_samples
    neg_scores = neg_scores.reshape(num_samples, num_neg)
    
    candidates = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=-1)
    target = torch.zeros(num_samples, dtype=torch.long) # positive sample is always in 0 idx
    
    metrics = evaluate(candidates, target, k_list=hits_k)
    metrics['loss'] = loss.item()
    return metrics

import re
import torch
from transformers import BertConfig

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
        print("[load] strict=True failed, falling back to exact-shape filter.")
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
    # Setup
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if args.load_checkpoint_and_eval is not None:
        is_best_ckpt = "_best" in os.path.basename(args.load_checkpoint_and_eval)
        results_save_path = args.load_checkpoint_and_eval.replace('_best.pth','.json').replace('checkpoints','results') if is_best_ckpt else args.load_checkpoint_and_eval.replace('.pth','.json').replace('checkpoints','results').replace('checkpoints','results') 
        training_results = json.load(open(results_save_path))
        # Append a terminal JSON row that ONLY has test_* columns (so your parser can find it)
        # model, ckpt = build_model_matching_ckpt(args, args.load_checkpoint_and_eval)
        # # import argparse
        ckpt = torch.load(args.load_checkpoint_and_eval, map_location="cpu")
        saved_args = ckpt.get("args")
        saved_args['load_checkpoint_and_eval'] = args.load_checkpoint_and_eval
        args.seed = saved_args['seed']
        if isinstance(saved_args, dict):
            args = argparse.Namespace(**saved_args)
    
    args.use_wandb = False
    set_seed(args.seed)
    
    model_save_path = os.path.join("stored", f"{args.experiment_name}_model.pth")
    personal_file = os.path.join(args.input_path, "personal.csv")

    graph_data, user_sequences_times, node_map = load_and_prepare_data(args)

    # Create a reverse map to translate final indices back to original UIDs for analysis
    idx_to_original_uid = {new_id: old_id for old_id, new_id in node_map.items()}

    # 2. Separate sequences and timestamps (your logic here is correct)
    # The keys `idx` are now the correct 0..N-1 indices.
    user_sequences = {
        idx: [event for event, timestamp in seq]
        for idx, seq in user_sequences_times.items()
    }

    user_event_times = {
        idx: [timestamp for event, timestamp in seq]
        for idx, seq in user_sequences_times.items()
    }

    # 3. Set up the tokenizer (this logic is correct)
    vocab = generate_vocab(args.csv_file, args.event_set)
    tokenizer = ProductRatingTokenizer(vocab)
    tokenizer.padding_side = 'left' 
    max_seq_tokens = args.max_events * 4 + 1


    results_dir = f"./{args.dataset}_results_seeded/" # if args.dataset == "brightkite" else  "./results_gowalla/"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir.replace("results", "checkpoints"), exist_ok=True)
    gnn_layer_config_str = args.gnn_layer_types.lower().replace(',', '-')
    if "XAT" in args.gnn_layer_types.split(',')[0]:
        args.encoder_compress = 'stack'
        json_name = (
        f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}{'_self-loop' if args.add_self_loops else ''}{'_qknorm' if args.qk_norm else ''}{'_pre-agg-proj' if args.pre_agg_proj else ''}{'_post-agg-proj' if args.post_agg_proj else ''}_{args.aggregation_mode}{'_compress-first' if args.compress_first else ''}_compress-dim{args.compress_dim}{'_msg-mode-gcn' if args.msg_mode == 'gcn' else ''}{'_layer-norm' if args.layer_norm else ''}{'_gcn-edge-weight' if args.gcn_edge_weight else ''}_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
        f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
        f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}.json"
        if not args.debug_subset else
        f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}{'_self-loop' if args.add_self_loops else ''}{'_qknorm' if args.qk_norm else ''}{'_pre-agg-proj' if args.pre_agg_proj else ''}{'_post-agg-proj' if args.post_agg_proj else ''}_{args.aggregation_mode}{'_compress-first' if args.compress_first else ''}_compress-dim{args.compress_dim}{'_msg-mode-gcn' if args.msg_mode == 'gcn' else ''}{'_layer-norm' if args.layer_norm else ''}{'_gcn-edge-weight' if args.gcn_edge_weight else ''}_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
        f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
        f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_debug{args.debug_subset}_seed{args.seed}.json"
        )
    else:
        json_name = (
        f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
        f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
        f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_seed{args.seed}.json"
        if not args.debug_subset else
        f"e2e_{gnn_layer_config_str}{args.gnn_num_layers}x{args.gnn_heads}_bert{args.bert_num_layers}x{args.bert_num_heads}_hidden-dim{args.bert_hidden_size}_"
        f"{args.encoder_compress}_nngb{args.num_neighbors}_nneg{args.num_negative_samples}_"
        f"bs{args.batch_size}_lr{args.lr}_epochs{args.num_epochs}_debug{args.debug_subset}_seed{args.seed}.json")
    results_save_path = os.path.join(results_dir,json_name)
    run_name = json_name.replace(".json","")
    project_name = f"{args.dataset}-link-prediction"
    if args.use_wandb:
        wandb.init(
                project=project_name,
                config=vars(args),
                name=run_name,
                dir="./wandb_runs"  # <-- Add this line
                
            )


    train_loader = LinkNeighborLoader(
        graph_data,
        num_neighbors=[args.num_neighbors] * args.gnn_num_layers,
        # num_neighbors=[args.num_neighbors] * len(model.gnn.layers),
        # For training, we predict the message-passing edges themselves (self-supervision)
        edge_label_index=graph_data.edge_index, 
        batch_size=args.batch_size,
        shuffle=True,
        # The loader will automatically create negative edges for each positive edge
        neg_sampling_ratio=args.train_neg_sampling_ratio, 
        num_workers=args.num_workers
    )

    # -- Model Initialization --
    bert_config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=args.bert_hidden_size,
        num_hidden_layers=args.bert_num_layers,
        num_attention_heads=args.bert_num_heads,
        intermediate_size=args.bert_hidden_size * 2,
        max_position_embeddings=max_seq_tokens,
    )


    
    if args.run_diagnostic_gcn:
        logger.info("--- RUNNING DIAGNOSTIC GCNWithTokenAggregation BASELINE ---")
        # This diagnostic model has a simpler constructor
        model = GCNWithTokenAggregation(
            bert_config=bert_config,
            gnn_hidden_channels=args.bert_hidden_size,
            gnn_num_layers=args.gnn_num_layers,
            gnn_dropout=args.gnn_dropout,
        ).to(device)
        # Update the experiment name to reflect the diagnostic run
        args.experiment_name += "_diag_gcn_agg"
    else:
        logger.info("--- RUNNING STANDARD JOINT MODEL ---")
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
    all_user_ids = range(graph_data.num_nodes)
    all_sequences_ids = []
    all_sequences_timestamps = [] 
    logger.info("Pre-tokenizing and masking all user sequences...")
    uid_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}

    for uid in tqdm(all_user_ids, desc="Tokenizing sequences"):
        # breakpoint()
        # seq = user_sequences[uid_to_idx[uid]] # Use empty list for users with no sequence
        seq = user_sequences.get(uid, [])
        timestamp = user_event_times.get(uid, [])
        try:
            timestamp_by_4 = list(itertools.chain.from_iterable([[timestamp[i]]* 4 for i in range(len(timestamp))])) + [timestamp[-1] if seq else 0]
        except Exception as e:
            print(len(timestamp))
            breakpoint()
        token_ids = []
        # We don't pad here, we pad in the model's encode function
        input_ids = tokenizer.encode(seq, max_length=max_seq_tokens, padding="max_length", truncation=True)
        attention_mask = [1 if token_id != tokenizer.vocab[tokenizer.pad_token] else 0 for token_id in input_ids]
        # Now, pad both to max length
        all_sequences_ids.append(input_ids)
        pad_length = max_seq_tokens - len(timestamp * 4) - 1
        try:
            pad_times = [timestamp[-1] if seq else 0] *  pad_length
        except Exception as e:
            print(len(timestamp))
            breakpoint()
        all_sequences_timestamps.append(pad_times + timestamp_by_4)

        

        

    # Convert to tensors and move to device
    # breakpoint()
    user_sequences_ids= torch.tensor(all_sequences_ids, dtype=torch.long).to(device)
    # user_sequences_labels = torch.tensor(all_sequences_labels, dtype=torch.long).to(device)
    user_sequences_masks = (user_sequences_ids != vocab["[PAD]"]).long().to(device)
    user_timestamps = torch.tensor(all_sequences_timestamps, dtype=torch.long).to(device)
    

    # criterion = nn.CrossEntropyLoss(ignore_index=-100)
    criterion = F.binary_cross_entropy_with_logits
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ----------------- EVAL-ONLY SHORT-CIRCUIT -----------------
    if args.load_checkpoint_and_eval:
        logger.info(f"--- Evaluation-only mode ---")
        logger.info(f"Loading checkpoint: {args.load_checkpoint_and_eval}")
        is_best_ckpt = "_best" in os.path.basename(args.load_checkpoint_and_eval)
        results_save_path = args.load_checkpoint_and_eval.replace('_best.pth','.json').replace('checkpoints','results') if is_best_ckpt else args.load_checkpoint_and_eval.replace('.pth','.json').replace('checkpoints','results').replace('checkpoints','results') 
        training_results = json.load(open(results_save_path))
        # Append a terminal JSON row that ONLY has test_* columns (so your parser can find it)
        # model, ckpt = build_model_matching_ckpt(args, args.load_checkpoint_and_eval)
        # import argparse
        # ckpt = torch.load(args.load_checkpoint_and_eval, map_location="cpu")
        # saved_args = ckpt.get("args")
        # saved_args['load_checkpoint_and_eval'] = args.load_checkpoint_and_eval
        # if isinstance(saved_args, dict):
        #     args = argparse.Namespace(**saved_args)
        # breakpoint()
        # 2) Load weights
        model, ckpt = build_model_matching_ckpt(args, args.load_checkpoint_and_eval)
        load_strict_or_filtered(model, ckpt)

        # 3) (Optional) sanity print
        # from the earlier compare function you already have:
        # compare_ckpt_to_model(model, ckpt_path, device=device)

        # 4) Move to device / eval and run test(...)
        model.to(device)
        model.eval()
        # _loaded_args, _loaded_test_metrics = load_checkpoint(args.load_checkpoint_and_eval, model, optimizer, device)
        # ckpt, report = safe_load_checkpoint(model, args.load_checkpoint_and_eval, device=device)
        # breakpoint()
        # Run TEST evaluation
        test_metrics = test(
            model, model.predictor, graph_data,
            args.eval_batch_size, device, args.num_neighbors, args.hits_k,
            user_sequences_ids, user_sequences_masks, user_timestamps,
            criterion, split="test"
        )
        logger.info("Eval-only Test Metrics | " + " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))

        # Decide prefix based on filename ("best_test_" if '_best' in filename)

        prefix = "best_test_" if is_best_ckpt else "test_"

        # Derive results json path from checkpoint (checkpoints -> results; .pth -> .json)
        results_json_from_ckpt = (
            args.load_checkpoint_and_eval
            .replace("checkpoints", "results")
            .replace("_best.pth", ".json")
            .replace(".pth", ".json")
        )
        # breakpoint()
        # Prefix keys and add metadata
        prefixed = {f"{prefix}{k.lower().replace('@','_at_')}": float(v) for k, v in test_metrics.items()}
        prefixed.update({
            "eval_only": True,
            "checkpoint_used": os.path.basename(args.load_checkpoint_and_eval),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })
        training_results.append(prefixed)
        # Append to JSON
        append_row_to_json_log(results_json_from_ckpt, prefixed)
        logger.info(f"Appended eval-only metrics to {results_json_from_ckpt}")
        return  # stop here; skip training

    initial_metrics_log = []
    args.num_negative_samples = graph_data.val_neg_edge_index.shape[1] // graph_data.val_edge_index.shape[1]
    num_batches = len(train_loader)
    model.eval()  # Set the model to evaluation mode
    total_initial_loss = 0.0
    train_begin_time = time.time()
    
    
    

    # with torch.no_grad():
    #     # --- COMPUTE INITIAL LOSS ON THE ENTIRE TRAINING SET ---
    #     logging.info("Calculating initial loss on the full training dataset...")
    #     # # # Use tqdm for a progress bar, similar to your training loop

    #     for batch in tqdm(train_loader, desc="Calculating Initial Loss"):
    #         batch.val_edge_index = None; batch.test_edge_index = None
    #         batch.val_neg_edge_index = None; batch.test_neg_edge_index = None
    #         batch = batch.to(device)
    #         # Forward pass
    #         gnn_out = model(user_sequences_ids, user_sequences_masks, batch, user_timestamps, training = False)
    #         # breakpoint()
    #         # gnn_out2 = model2(user_sequences_ids, user_sequences_masks, batch)
    #         # breakpoint()
    #         scores = model.predictor(gnn_out[batch.edge_label_index[0]], gnn_out[batch.edge_label_index[1]]).squeeze()
    #         # --- Calculate Losses ---
    #         # 1. Link Prediction Loss
    #         # link_loss = loss_fn(link_preds, batch.edge_label.float())
    #         # breakpoint()
    #         link_loss = criterion(scores, batch.edge_label.float())
    #         # Accumulate the loss
    #         total_initial_loss += link_loss.item()

    # # Calculate the average loss over all batches
    average_initial_loss = total_initial_loss / num_batches
    epoch_time = time.time() - train_begin_time
    logging.info(f"Initial Average Training Loss (Full Dataset): {average_initial_loss:.4f}")
    gpu_mem_allocated = 0
    gpu_mem_peak = 0
    # Only log if CUDA is available and in use
    if device.type == 'cuda':
        # Get current and peak memory in MB
        gpu_mem_allocated = round(torch.cuda.memory_allocated(device) / (1024**2), 2)
        gpu_mem_peak = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        # Reset the peak memory "high-water mark" for the next epoch
        torch.cuda.reset_peak_memory_stats(device)
            
    # --- Evaluate on the Validation Set ---
    eval_begin_time = time.time()
    initial_val_metrics = test(model, model.predictor, graph_data, args.eval_batch_size, device, args.num_neighbors, args.hits_k,  user_sequences_ids, user_sequences_masks, user_timestamps, criterion, split="val")
    eval_time = time.time() - eval_begin_time
    val_mrr = initial_val_metrics.get("MRR", 0.0)
    val_link_loss = initial_val_metrics.get("loss", 0.0) # Get initial val loss for the progress bar
    log_msg_val = f"Initial Pre-Training Metrics (Val)   | Loss: {val_link_loss:.4f}, MRR: {val_mrr:.4f}"
    for k in args.hits_k:
        log_msg_val += f", Hits@{k}: {initial_val_metrics.get(f'Hits@{k}', 0.0):.4f}"
    logging.info(log_msg_val)
    # breakpoint()
    if args.use_wandb:
        wandb_log = {
            'train_loss': average_initial_loss,
        }
        if initial_val_metrics:
            # Flatten the metrics dict for better plotting
            for key, value in initial_val_metrics.items():
                wandb_log[f"val_{key}"] = value
        
        wandb.log(wandb_log, step=0)
    epoch0_row = {'epoch': 0, 'train_loss': average_initial_loss, "evaluation_metrics": initial_val_metrics, "epoch_time": epoch_time, "eval_time": eval_time, "gpu_mem_allocated_mb": gpu_mem_allocated,
        "gpu_mem_peak_mb": gpu_mem_peak
    }
    initial_metrics_log.append(epoch0_row)

    training_results, best_model_state, best_metric, best_time = train_model(
        model,
        train_loader,
        graph_data, 
        user_sequences_ids, 
        user_sequences_masks, 
        user_timestamps, optimizer, 
        tokenizer, device,
        criterion, 
        args, 
        initial_metrics_log=initial_metrics_log, 
        log_path = results_save_path,
        run_name = run_name)


    if training_results:
        with open(results_save_path.replace('.csv','.json'), 'w') as f:
                json.dump(training_results, f, indent=4)
        checkpoint_path = results_save_path.replace("results","checkpoints").replace('.json', '.pth')
        best_model_path = checkpoint_path.replace('.pth', '_best.pth')
    
        total_elapsed_time = sum(epoch_data.get('epoch_time', 0) for epoch_data in training_results)
        final_metrics = training_results[-1]
        
        # Call the save function with the JointModel instance
        save_checkpoint(
            filepath=checkpoint_path,
            model=model, # model is the instance of JointModel
            optimizer=optimizer,
            args=args,
            elapsed_time=total_elapsed_time,
            test_metrics=final_metrics
        )
                # --- 2. Save the BEST model state locally ---
        if best_model_state: # Check if a best state was captured
            # Create a new filepath for the best model checkpoint
            best_checkpoint_path = results_save_path.replace("results","checkpoints").replace('.json', '_best.pth')
            
            # Temporarily load the best state into the model object
            logging.info("Loading best model state for local checkpointing...")
            model.load_state_dict(best_model_state)
            
            # Call save_checkpoint again, which now saves the best state
            save_checkpoint(
                filepath=best_checkpoint_path,
                model=model, # model now holds the best weights
                optimizer=optimizer,
                args=args,
                elapsed_time=total_elapsed_time, # Note: time/metrics are still from the final epoch
                test_metrics=final_metrics
            )
        
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Joint Training of Sequence and Graph Models")
    
    # Data Args
    parser.add_argument("--dataset", type=str, choices=['clothing','movies','electronics','musical'], default='musical')
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
    parser.add_argument('--num_neighbors', type=int, default=2)
    parser.add_argument('--add_self_loops', action='store_true')
    parser.add_argument('--aggregation_mode', type=str, default='fixed')
    parser.add_argument('--compress_dim', type=int, default=4)
    parser.add_argument('--qk_norm', action='store_true')
    parser.add_argument('--compress_first', action='store_true')
    parser.add_argument('--msg_mode',type=str, choices=['gat','gcn'], default='gat',help="Test if TokenCrossAttn is equivalent to gat/ gcn when compressed first")
    parser.add_argument('--gcn_edge_weight', action='store_true')
    # Training Args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256 * 4,
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

    parser.add_argument(
        '--load_checkpoint_and_eval',
        type=str,
        default=None,
        help="If set, load this checkpoint, evaluate on TEST split, and append results to the JSON log (no training)."
    )
    args = parser.parse_args()
    args.hits_k = [int(k) for k in args.hits_k.split(',')]

    args.csv_file = f"../data/tasks/amazon-{args.dataset}/coreview_prediction/personal_observed.csv"
    args.input_path = f"../data/tasks/amazon-{args.dataset}/coreview_prediction/"
    main(args)
