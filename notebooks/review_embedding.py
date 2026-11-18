import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv, TransformerConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import os
import time
import random
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn as nn
# from brightkite import AttentionCompressor, TokenCrossAttnConv #, TokenCrossAttnConvGCN
from xat import AttentionCompressor, TokenCrossAttnConv
from tqdm import tqdm
import pickle
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
import json # <-- Added for local logging
from numpy.lib.format import open_memmap


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
    print(f"Saved data splits to: {split_file}")
    return split_file


def load_data_splits(split_dir, dataset_name, seed, device):
    """
    Load pre-saved train/val/test splits from disk.
    Returns the splits if found, None otherwise.
    """
    split_file = os.path.join(split_dir, f'{dataset_name}_splits_seed{seed}.pt')
    
    if os.path.exists(split_file):
        print(f"Loading pre-saved data splits from: {split_file}")
        splits = torch.load(split_file, map_location=device)
        return splits['train_mask'], splits['val_mask'], splits['test_mask']
    else:
        print(f"No pre-saved splits found at: {split_file}")
        return None, None, None



class LocalLogger:
    """
    A simple logger to save metrics to a local JSON file,
    mimicking the basic functionality of wandb.
    """
    def __init__(self, project_name, run_name, log_dir='logs_delta'):
        """
        Initializes the logger, creating directories and setting up the file path.
        If the log file for the run_name already exists, it marks the run to be skipped.
        
        Args:
            project_name (str): The name of the project, used as a directory name.
            run_name (str): The name of the specific run, used as the JSON filename.
            log_dir (str): The root directory for all logs.
        """
        self.log_data = []
        self._t0 = time.time()
        self._skip_experiment = False
        log_path = os.path.join(log_dir, project_name)
        os.makedirs(log_path, exist_ok=True)
        
        # Ensure the filename ends with .json
        if not run_name.endswith('.json'):
            run_name += '.json'
        self.log_file = os.path.join(log_path, run_name)

        # Check if the log file already exists
        if os.path.exists(self.log_file):
            print(f"Log file already exists: {self.log_file}. Skipping experiment.")
            self._skip_experiment = True
        else:
            print(f"Local logger initialized. Metrics will be saved to: {self.log_file}")

    @property
    def skip(self):
        """Returns True if the experiment should be skipped."""
        return self._skip_experiment

    def log(self, data: dict):
        """
        Stores a dictionary of metrics if the experiment is not being skipped.
        
        Args:
            data (dict): A dictionary containing metric names and their values.
        """
        if self.skip:
            return
        self.log_data.append(data)

    def finish(self):
        """
        Writes all stored log data to the JSON file if the experiment was not skipped.
        """
        if self.skip:
            return
        # Stamp end-to-end wall time (logger creation -> finish) for the cost comparison.
        self.log_data.append({"wall_sec_total": time.time() - self._t0})
        # Convert numpy scalars/arrays to plain Python types for clean JSON.
        def _to_serializable(obj):
            # Handle numpy scalar types (e.g., np.float64, np.int64)
            if isinstance(obj, np.generic):
                return obj.item()
            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle dictionaries recursively
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            # Handle lists/tuples recursively
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        serializable_data = _to_serializable(self.log_data)
        with open(self.log_file, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        print(f"Successfully saved log to {self.log_file}")



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
    



def recall_fraction(labels: np.ndarray, probs: np.ndarray, top_frac=0.01) -> float:
    """Recall fraction in top scoring x % of samples
    Args:
        labels (np.ndarray): labels
        probs (np.ndarray): positive probabilities
        top_frac (float, optional): top x fraction of accounts to measure recall. Defaults to 0.01.
    Returns:
        float: recall fraction
    """

    if top_frac <= 0 or top_frac > 1:
        raise ValueError("top_frac must be in range (0,1]")

    return labels[np.argsort(probs)[-int(len(probs) * top_frac) :]].sum() / labels.sum()


class ProductRatingTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.cls_token = "[CLS]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"

    def tokenize_product_rating(self, event):
        # Expect the event string format to be "ASIN:rating", e.g., "B0001FTVD6:5"
        if ":" not in event:
            return [self.unk_token]
        asin, rating = event.split(":", 1)
        # Pad the ASIN with zeros to ensure length 10.
        asin = asin.zfill(10)
        token1 = f"asin1-{asin[:4]}"
        token2 = f"asin2-{asin[4:7]}"
        token3 = f"asin3-{asin[7:10]}"
        tokens = [token1, token2, token3, f"rate-{rating}"]
        return tokens

    def encode(self, tokens, max_length=None, padding='max_length', truncation=True):
        processed_tokens = []
        for token in tokens:
            # If the token is a special token, leave it unchanged.
            if token in [self.cls_token, self.pad_token, self.unk_token, self.mask_token]:
                processed_tokens.append(token)
            else:
                # Each event is tokenized into exactly 4 tokens.
                processed_tokens.extend(self.tokenize_product_rating(token))
        # Append the [CLS] token at the end.
        tokens_with_special = processed_tokens + [self.cls_token]
        ids = [self.vocab.get(tok, self.vocab[self.unk_token]) for tok in tokens_with_special]
        if truncation and max_length is not None:
            # Keep the rightmost tokens (most recent events and the [CLS] token)
            ids = ids[-max_length:]
        if padding == 'max_length' and max_length is not None:
            ids = [self.vocab[self.pad_token]] * (max_length - len(ids)) + ids
        return ids

    def decode(self, ids):
        return [self.inv_vocab.get(i, self.unk_token) for i in ids]


def top_precision(labels: np.ndarray, probs: np.ndarray, top_frac=0.01) -> float:
    """Precision in top scoring x % of samples
    Args:
        labels (np.ndarray): labels
        probs (np.ndarray): positive probabilities
        top_frac (float, optional): top x fraction of accounts to measure precision. Defaults to 0.01.
    Returns:
        float: top precision
    """
    return labels[np.argsort(probs)[-int(len(probs) * top_frac) :]].mean()


def max_f1_score(labels: np.ndarray, probs: np.ndarray, **kwargs):
    """Maximum f1 score across all thresholds
    Args:
        labels (np.ndarray): labels
        probs (np.ndarray): positive probabilities
    Returns:
        float: best f1 score with best threshold
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs, **kwargs)
    return np.nanmax(2 * recall * precision / (recall + precision))



class MLPClassifier(nn.Module):
    """A flexible MLP classifier with a variable number of hidden layers."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        
        if num_layers < 2:
            self.layers = nn.Linear(in_channels, out_channels)
        else:    
            layers = []
            # Input Layer
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            
            # Hidden Layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
                
            # Output Layer
            layers.append(nn.Linear(hidden_channels, out_channels))
            
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SentenceClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, args = None):
        super().__init__()
        # A single linear layer is enough for classification
        # self.classifier = nn.Linear(in_channels, out_channels)
        # breakpoint()
        self.rating_embedding = nn.Embedding(6, args.rating_emb_dim) 
        combined_in_channels = in_channels + args.rating_emb_dim
        self.classifier = MLPClassifier(combined_in_channels , combined_in_channels, out_channels, num_layers=args.clf_layers)

    def forward(self, x,  edge_index, pad_mask, ratings, training = None):
        # 1. Perform masked mean pooling on the event embeddings
        rating_embeds = self.rating_embedding(ratings)
        x = torch.cat([x, rating_embeds], dim=-1)
        mask = pad_mask.unsqueeze(-1)
        x = x * mask
        sum_embeddings = torch.sum(x, dim=1)
        num_tokens = torch.sum(pad_mask, dim=1).unsqueeze(-1)
        pooled_embedding = sum_embeddings / num_tokens.clamp(min=1e-9)
        # breakpoint()
        
        # 2. Classify the resulting user embedding
        logits = self.classifier(pooled_embedding)
        return F.log_softmax(logits, dim=1)

import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (no trainable params).
    Expects input shape (N, T, D).
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even idx
        pe[:, 1::2] = torch.cos(position * div_term)  # odd idx
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, D)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        add_positional_encoding: bool = True,
        max_len: int = 1000,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.add_pe = add_positional_encoding

        # 1) Decide the Transformer model dimension and (optional) input projection
        self.project_in = None
        model_dim = in_dim
        if out_dim is not None and out_dim != in_dim:
            self.project_in = nn.Linear(in_dim, out_dim)  # project into model space
            model_dim = out_dim

        # 2) Create PE **in model space**
        self.pe = PositionalEncoding(model_dim, dropout=0.0, max_len=max_len) if self.add_pe else None

        # 3) Build the encoder **with model_dim**
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.model_dim = model_dim

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- PLACE THE BLOCK RIGHT HERE ----
        if self.project_in is not None:
            x = self.project_in(x)     # (N, T, model_dim)
        if self.add_pe:
            x = self.pe(x)             # add PE in model space

        # (Optional) zero pads explicitly if you want pads to stay exactly 0 after PE:
        # x = x * pad_mask.unsqueeze(-1)

        # PyTorch expects True where tokens are PADs; we have True for real tokens
        key_padding_mask = ~pad_mask   # (N, T) True at pads

        x_enc = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (N, T, model_dim)

        # masked mean pool
        mask3 = pad_mask.unsqueeze(-1)
        pooled = (x_enc * mask3).sum(dim=1) / mask3.sum(dim=1).clamp(min=1e-9)
        return x_enc, pooled



class Bridge(nn.Module):
    """
    Bridge = SequenceEncoder (temporal Transformer over events) --> GraphModel.

    - SequenceEncoder enriches per-event embeddings via self-attention across time.
    - The resulting token embeddings are passed to GraphModel, which may further
      attend/compress across events (e.g., via TokenXAT) and then propagate over the user graph.

    Inputs to forward:
      x: (N, T, D) SBERT embeddings
      edge_index: PyG edges (2, E)
      pad_mask: (N, T) bool; True=real, False=pad
      ratings: (N, T) long; per-event ratings (0..5 with 0 as pad)
      training: bool
    """
    def __init__(
        self,
        in_channels: int,
        args,
        gnn_hidden_channels: int,
        dropout: float,
    ):
        super().__init__()


        seq_in_dim = in_channels + args.rating_emb_dim   # because you feed ratings before the Transformer
        self.seq = SequenceEncoder(
            in_dim=seq_in_dim,
            nhead=args.tr_nhead,
            num_layers=args.tr_layers,
            dim_feedforward=args.tr_ffn_dim,
            dropout=args.tr_dropout,
            add_positional_encoding=not args.tr_no_pos_enc,
            max_len=args.max_events,
            out_dim=in_channels
        )

        # 2) Graph part: GraphModel expects `in_channels` equal to token feature dim
        # It also adds rating embeddings internally.
        self.graph = GraphModel(
            in_channels=in_channels,
            hidden_channels=gnn_hidden_channels,
            gnn_layer_types=args.gnn_layer_types,
            dropout=dropout,
            args=args,
            include_ratings=False # always include ratings in Bridge
        )

        # Supervised warm-up head over the pooled sequence-encoder output. Used ONLY to pretrain the
        # encoder on the task before freezing it (supervised-frozen control); unused end-to-end.
        self.warmup_head = MLPClassifier(in_channels, in_channels, 2, num_layers=args.clf_layers)

    def forward(self, x, edge_index, user_timestamps=None, pad_mask=None, ratings=None, training=True, mode='graph'):
        # Sequence encoder; keep token-wise outputs for GraphModel
        rating_emb = self.graph.rating_embedding(ratings)         # (N,T,Dr)
        x_tokens_in = torch.cat([x, rating_emb], dim=-1)          # (N,T,384+Dr)
        x_tokens, pooled = self.seq(x_tokens_in, pad_mask)        # single call
        if mode == 'warmup':
            # supervised-frozen phase A: classify from the pooled encoder output, no graph.
            return F.log_softmax(self.warmup_head(pooled), dim=1)
        out = self.graph(x_tokens, edge_index, pad_mask=pad_mask, ratings=ratings, training=training)
        return out

    def freeze_encoder(self):
        """Freeze the sequence encoder (params + eval mode) for the supervised-frozen control."""
        self.seq.eval()
        for p in self.seq.parameters():
            p.requires_grad_(False)


class GraphModel(nn.Module):
    """
    A flexible GNN model that can stack different layer types dynamically.
    - If the first layer is TokenXAT, x remains 3D until the end of the stack, then flattened.
    - Otherwise we mean-pool tokens to 2D before the first 2D GNN layer.
    - Residual path matches the final feature dimensionality (3D->flattened or 2D).
    """
    def __init__(self, in_channels, hidden_channels, gnn_layer_types: str,
                 dropout, args=None, include_ratings: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.include_ratings = include_ratings
        self.dropout = dropout

        # Always build the table; only concat if include_ratings=True
        self.rating_embedding = nn.Embedding(6, args.rating_emb_dim)

        # Current channel width of per-token features entering the 1st layer
        current_in_channels = in_channels + args.rating_emb_dim if self.include_ratings else in_channels

        # Build layers
        gnn_types = gnn_layer_types.split(",")
        self.first_layer_type = gnn_types[0]
        self.residual_compressor = None

        for i, model_type in enumerate(gnn_types):
            if model_type == 'TokenXAT':
                # TokenCrossAttnConv keeps token grid; produces k tokens with same per-token dim (heads concat)
                layer = TokenCrossAttnConv(
                    in_channels=current_in_channels,
                    out_channels=current_in_channels // args.heads,  # per-head, concat -> current_in_channels
                    heads=args.heads,
                    num_agg_queries=args.compress_dim,
                    pre_agg_proj=args.pre_agg_proj,
                    post_agg_proj=args.post_agg_proj,
                    compress_mode=args.compress_mode,
                    add_self_loops=args.add_self_loops,
                    use_qk_norm=args.qk_norm,
                    compress_first=args.compress_first,
                    aggregation_mode=args.aggregation_mode,
                    residual=args.residual,
                    num_tokens=args.max_events
                )
                self.layers.append(layer)

                # After TokenXAT, if we later flatten, the flattened width increases by factor k
                # Track the flattened width accordingly:
                current_in_channels = layer.k * current_in_channels

                # Expose the compressor used on the residual branch (provided by your TokenXAT impl)
                if i == 0:
                    # If the model starts with TokenXAT, we keep a dedicated residual compressor
                    # so the residual features match the flattened shape later.
                    self.residual_compressor = layer.res  # your implementation attribute

            elif model_type == 'GCN':
                layer = GCNConv(current_in_channels, current_in_channels)
                self.layers.append(layer)

            elif model_type == 'GraphSAGE':
                layer = SAGEConv(current_in_channels, current_in_channels)
                self.layers.append(layer)

            elif model_type == 'GAT':
                # out_channels per head, concat -> current_in_channels preserved
                layer = GATConv(current_in_channels, current_in_channels // args.heads,
                                heads=args.heads, dropout=dropout, residual=False)
                self.layers.append(layer)

            elif model_type == 'GATv2':
                layer = GATv2Conv(current_in_channels, current_in_channels // args.heads,
                                  heads=args.heads, dropout=dropout, residual=False)
                self.layers.append(layer)

            elif model_type == 'TransformerConv':
                layer = TransformerConv(current_in_channels, current_in_channels // args.heads,
                                        heads=args.heads, dropout=dropout)
                self.layers.append(layer)
            else:
                raise ValueError(f"Unsupported GNN layer type: {model_type}")

        # Final MLP: its input width equals the feature width after the stack:
        # - If the stack contains TokenXAT, we flatten tokens -> (N, current_in_channels)
        # - Else, features remain 2D as current_in_channels
        self.classifier = MLPClassifier(current_in_channels, in_channels, 2, num_layers=args.clf_layers)

    def forward(self, x, edge_index, user_timestamps=None, pad_mask=None, ratings=None, training=True):
        """
        x: (N, T, C) if first layer is TokenXAT; otherwise we pool to (N, C) before the first 2D layer.
        pad_mask: (N, T) with True for real tokens, False for pads.
        """
        # Optional rating concat at the token level
        if self.include_ratings:
            rating_embeds = self.rating_embedding(ratings)   # (N,T,Dr)
            x = torch.cat([x, rating_embeds], dim=-1)        # (N,T,C + Dr)

        # Residual branch input is the token grid as of now
        residual_input = x  # (N,T,feat_token)

        # --------- Residual path construction ----------
        if self.residual_compressor is not None:
            # FIRST LAYER is TokenXAT: build residual by compressing the (masked) token grid
            # IMPORTANT: zero out pads before flattening to avoid pad-noise
            mask3 = pad_mask.unsqueeze(-1)                                  # (N,T,1)
            residual_masked = residual_input * mask3                        # zero pad tokens
            N, T, Ctok = residual_masked.shape
            residual_features = self.residual_compressor(residual_masked.reshape(N, -1))  # (N, k*Ctok) expected
        else:
            # FIRST LAYER is 2D (GCN/SAGE/GAT/TransformerConv): residual is masked mean over tokens
            mask3 = pad_mask.unsqueeze(-1)                                  # (N,T,1)
            residual_masked = residual_input * mask3
            denom = mask3.sum(dim=1).clamp(min=1e-9)                        # (N,1)
            residual_features = residual_masked.sum(dim=1) / denom          # (N, Ctoken)

        # --------- Message passing stack ----------
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)

            if isinstance(layer, TokenCrossAttnConv):
                # TokenXAT keeps 3D token grid internally
                x = layer(x, edge_index, mask=pad_mask, training=training)
            else:
                if i == 0 and self.first_layer_type != 'TokenXAT':
                    # Convert token grid to user 2D features by masked mean BEFORE first 2D layer
                    mask3 = pad_mask.unsqueeze(-1)
                    x2d = (x * mask3).sum(dim=1) / mask3.sum(dim=1).clamp(min=1e-9)  # (N, C)
                    x = layer(x2d, edge_index)
                else:
                    x = layer(x, edge_index)

            if not is_last:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        # If the last output is 3D (TokenXAT), flatten tokens -> (N, k*C)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        # Ensure residual has the same shape (defensive assert helps catch mismatches early)
        if residual_features.dim() > 2:
            residual_features = residual_features.reshape(residual_features.size(0), -1)
        assert x.shape == residual_features.shape, \
            f"Residual shape {residual_features.shape} != main path shape {x.shape}. "\
            f"Check TokenXAT.res output width and masking."

        # Residual merge and classification
        combined = x + residual_features
        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=1)


# --- 1. Define the Graph Neural Network Model ---
class GNN(torch.nn.Module):
    """
    A simple two-layer GraphSAGE model for node classification.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def build_graph_data(edges_df, uid_map, x, ratings, y, mask,args):
    """
    Builds a PyG Data object from pre-processed tensors and dataframes.
    This is where the edge_index is correctly mapped.
    """
    # 1. Map the original UIDs in the edge list to their new 0-based indices
    source_nodes = edges_df['uid'].map(uid_map)
    target_nodes = edges_df['other_uid'].map(uid_map)

    # 2. Create a new DataFrame and drop edges with users not in the node set
    mapped_edges = pd.DataFrame({'source': source_nodes, 'target': target_nodes})
    mapped_edges.dropna(inplace=True)

    # 3. Create the final edge_index tensor
    edge_index = torch.tensor(mapped_edges.values, dtype=torch.long).t().contiguous()

    # 4. Add a sanity check to prevent CUDA errors
    if edge_index.numel() > 0:
        assert edge_index.max() < x.size(0), "FATAL: edge_index contains an out-of-bounds node ID!"
    
    # 5. Create and return the PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y, mask=mask, ratings=ratings)
    return data

# --- 3. Training and Evaluation Functions ---
def train(model, data, optimizer, loss_fn, freeze_seq=False):
    model.train()
    if freeze_seq:
        model.seq.eval()  # keep the frozen sequence encoder deterministic during graph-phase training
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, pad_mask=data.mask, ratings= data.ratings, training = True)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    # Calculate training accuracy for the batch
    with torch.no_grad():
        pred = out.argmax(dim=1)
        proba = out.exp()[:, 1]
        acc = (pred == data.y).sum() / len(data.y)
        roc_auc = roc_auc_score(data.y.cpu(), proba.cpu())
        
    return loss.item(), acc.item(), roc_auc

@torch.no_grad()
def test(model, loader, device, loss_fn, return_preds=False):
    """
    Evaluates the model on a given data loader (validation or test).
    """
    model.eval()

    all_preds = []
    all_probas = []
    all_labels = []
    all_nids = []
    total_loss = 0
    total_nodes = 0

    # Process data in mini-batches
    for batch in loader:
        batch = batch.to(device)
        
        # Get model output for all nodes in the computational graph
        out = model(batch.x, batch.edge_index, pad_mask=batch.mask, ratings = batch.ratings, training=False)
        
        # We only evaluate the predictions for the seed nodes of the batch
        batch_size = batch.batch_size
        out = out[:batch_size]
        labels = batch.y[:batch_size]

        # Calculate loss for the batch
        loss = loss_fn(out, labels)
        total_loss += loss.item() * batch_size
        total_nodes += batch_size

        # Get predictions and probabilities for this batch
        pred = out.argmax(dim=1)
        proba = out.exp()[:, 1] # Probability of the positive class

        # Store results from the batch
        all_preds.append(pred.cpu())
        all_probas.append(proba.cpu())
        all_labels.append(labels.cpu())
        all_nids.append(batch.n_id[:batch_size].cpu())

    # After iterating through all batches, concatenate the results
    final_preds = torch.cat(all_preds)
    final_probas = torch.cat(all_probas).numpy()
    final_labels = torch.cat(all_labels).numpy().astype(int)

    # Calculate metrics over the entire validation/test set
    avg_loss = total_loss / total_nodes if total_nodes > 0 else 0
    acc = (final_preds.numpy() == final_labels).mean()
    roc_auc = roc_auc_score(final_labels, final_probas)
    
    precision, recall, _ = precision_recall_curve(final_labels, final_probas)
    pr_auc = auc(recall, precision)
    recall_at_10_percent = recall_fraction(final_labels, final_probas, top_frac=0.1)
    precision_at_10_percent = top_precision(final_labels, final_probas, top_frac=0.1)
    max_f1 = max_f1_score(final_labels, final_probas)
    
    metrics = {
        'acc': acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'recall_at_10_percent': recall_at_10_percent,
        'precision_at_10_percent': precision_at_10_percent,
        'max_f1': max_f1,
        'loss': avg_loss
    }
    
    if return_preds:
        final_nids = torch.cat(all_nids).numpy()
        return metrics, final_preds, final_probas, final_labels, final_nids
    return metrics, final_preds


import os, pickle, numpy as np
from tqdm import tqdm

def _embedding_dim_from_model(model, fallback=384):
    if hasattr(model, 'get_sentence_embedding_dimension'):
        try:
            return int(model.get_sentence_embedding_dimension())
        except Exception:
            pass
    # robust fallback: probe one encode
    try:
        v = model.encode(['probe'], convert_to_numpy=True, show_progress_bar=False)
        return int(v.shape[-1])
    except Exception:
        return int(fallback)

def process_and_embed_user_events(df, sbert_model, max_len, data_dir,
                                  dtype='float16', batch_size=64):
    """
    Streamed & memory-safe:
      - writes embeddings to a numeric .npy (memmap) as we go
      - saves masks/labels as small .npy
      - loads with allow_pickle=False
    """
    os.makedirs(data_dir, exist_ok=True)
    emb_path  = os.path.join(data_dir, 'all_embeddings.npy')   # (N, max_len, D)
    mask_path = os.path.join(data_dir, 'all_masks.npy')        # (N, max_len) bool
    label_path= os.path.join(data_dir, 'all_labels.npy')       # (N,)
    dict_path = os.path.join(data_dir, 'all_uid_map.pkl')

    cache_paths = [emb_path, mask_path, label_path, dict_path]

    def _delete_corrupt_cache():
        for path in cache_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                continue

    # ---------- Fast path: load existing (verify numeric) ----------
    if all(os.path.exists(p) for p in cache_paths):
        print("Loading pre-processed 'all' data via memory-mapping...")
        rebuild_required = False
        try:
            emb = np.load(emb_path, mmap_mode='r', allow_pickle=False)
        except ValueError as e:
            if "pickled" in str(e).lower():
                print("Cached embeddings appear pickled; deleting cache and rebuilding...")
                rebuild_required = True
            else:
                raise
        else:
            if emb.dtype == object:
                print("Cached embeddings have dtype=object; deleting cache and rebuilding...")
                rebuild_required = True
        if not rebuild_required:
            try:
                masks  = np.load(mask_path, mmap_mode='r', allow_pickle=False)
                labels = np.load(label_path, mmap_mode='r', allow_pickle=False)
                with open(dict_path, 'rb') as f:
                    uid_to_idx = pickle.load(f)
            except (ValueError, pickle.UnpicklingError) as e:
                print(f"Cached ancillary arrays are invalid ({e}); deleting cache and rebuilding...")
                rebuild_required = True
            else:
                return emb, masks, labels, uid_to_idx

        if rebuild_required:
            _delete_corrupt_cache()

    print("No valid cache found. Processing 'all' data from scratch...")

    # Lazy: construct the SBERT model only now (cache hits already returned above), so runs with a
    # warm embedding cache never need the model file at all.
    if not isinstance(sbert_model, SentenceTransformer):
        sbert_model = sbert_model()

    # ---------- Prepare groups & shapes ----------
    grouped = df.groupby('uid', sort=False)
    uids    = list(grouped.groups.keys())
    N       = len(uids)
    D       = _embedding_dim_from_model(sbert_model, fallback=384)

    # dtype handling
    dtype_np = np.float16 if str(dtype) == 'float16' else np.float32

    # ---------- Write to temp memmap, then atomically rename ----------
    tmp_emb_path = emb_path + '.tmp'
    if os.path.exists(tmp_emb_path):
        os.remove(tmp_emb_path)
    emb_mm = open_memmap(tmp_emb_path, mode='w+', dtype=dtype_np, shape=(N, max_len, D))

    masks  = np.zeros((N, max_len), dtype=bool)
    labels = np.zeros((N,), dtype=np.int8)
    uid_to_idx = {}

    for i, uid in enumerate(tqdm(uids, desc="Embedding users")):
        g = grouped.get_group(uid).sort_values('timestamp', ascending=True)

        reviews = g['reviewText'].astype(str).fillna('').tolist()[-max_len:]
        T = len(reviews)
        pad = max_len - T

        # Encode only the real reviews (skip padded ''), paste to the right end
        user_embs = np.zeros((max_len, D), dtype=dtype_np)
        if T > 0:
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                b = sbert_model.encode(
                    reviews[start:end],
                    batch_size=min(batch_size, end - start),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )  # float32
                if dtype_np == np.float16:
                    b = b.astype(np.float16, copy=False)
                user_embs[pad + start : pad + end] = b  # right-aligned

        emb_mm[i] = user_embs
        masks[i, pad:] = True if T > 0 else False
        labels[i] = int(g['fraud_label'].iloc[0])
        uid_to_idx[uid] = i

        if (i + 1) % 500 == 0:
            emb_mm.flush()

    emb_mm.flush()
    del emb_mm  # CLOSE the memmap before reading

    # Atomic move into place
    os.replace(tmp_emb_path, emb_path)

    # Save small arrays & dict
    np.save(mask_path, masks)
    np.save(label_path, labels)
    with open(dict_path, 'wb') as f:
        pickle.dump(uid_to_idx, f)

    embeddings_array = np.load(emb_path, mmap_mode='r', allow_pickle=False)
    masks_array      = np.load(mask_path, mmap_mode='r', allow_pickle=False)
    labels_array     = np.load(label_path, mmap_mode='r', allow_pickle=False)
    return embeddings_array, masks_array, labels_array, uid_to_idx


def prepare_user_sequences(df, uid_map, max_len):
    """
    Prepares rating sequences, masks, and labels for all users. This is fast and
    does not need caching.
    """
    print("Preparing user sequences (ratings, masks, labels)...")
    # Pre-allocate numpy arrays for efficiency
    num_users = len(uid_map)
    ratings_array = np.zeros((num_users, max_len), dtype=int)

    grouped = df.groupby('uid')
    for user_id, user_df in tqdm(grouped, desc="Processing user sequences"):
        if user_id in uid_map:
            idx = uid_map[user_id]
            user_df = user_df.sort_values('timestamp', ascending=True).tail(max_len)
            ratings = user_df['rating'].tolist()
            num_events = len(ratings)
            
            # Place data at the end of the sequence
            ratings_array[idx, -num_events:] = ratings
            
    return ratings_array

# --- 4. Main Execution Block ---
def main(args):
    print("--- GNN Training for Fraud Detection from CSV Splits ---")
    set_seed(args.seed)
    # --- NEW: Generate descriptive run name for W&B ---
    gnn_layer_config_str = args.gnn_layer_types.lower().replace(',', '-')
    # A simplified name for non-XAT models
    json_name = (
        f"{'bridge_' if args.bridge else ''}{gnn_layer_config_str}_heads{args.heads}_nngb{args.num_neighbors}_clf{args.clf_layers}_"
        f"lr{args.lr}_bs{args.batch_size}_epochs{args.epochs}_seed{args.seed}"
    )
    if "XAT" in args.gnn_layer_types.split(',')[0]:
        # A more detailed name for XAT models
        json_name = (
            f"{'bridge_' if args.bridge else ''}{gnn_layer_config_str}_heads{args.heads}_nngb{args.num_neighbors}"
            f"{'_self-loop' if args.add_self_loops else ''}{'_qkn' if args.qk_norm else ''}"
            f"_{args.compress_mode}{'_cf' if args.compress_first else ''}"
            f"_cdim{args.compress_dim}"
            f"{'_pre-agg-proj' if args.pre_agg_proj else ''}"
            f"{'_post-agg-proj' if args.pre_agg_proj else ''}"
            f"_agg-mode-{args.compress_mode}"
            f"_msg-mode-{args.aggregation_mode}"
            f"{'_residual' if args.residual else ''}"
            f"{'_ln' if args.layer_norm else ''}"
            f"_clf{args.clf_layers}_lr{args.lr}_bs{args.batch_size}_epochs{args.epochs}_seed{args.seed}"
        )
    
    if args.seq_only:
        json_name = f"seq_only_clf{args.clf_layers}_lr{args.lr}_bs{args.batch_size}_epochs{args.epochs}_seed{args.seed}"

    run_name = json_name
    if args.freeze_encoder and args.bridge:
        run_name = f"supfrozen{args.encoder_warmup_epochs}_{json_name}"
    project_name = f'fraud-{args.dataset}-{args.suffix}'
    logger = LocalLogger(project_name=project_name, run_name=run_name, log_dir=args.log_dir)
    if logger.skip:
        print("Exiting script because the experiment is being skipped.")
        return # This will stop the main function from continuing




    # --- Load Data Splits from CSV ---
    print("Step 1: Loading pre-processed CSV data splits...")

    script_dir = os.path.dirname(__file__)

    # Construct an absolute path to the data directory
    data_dir = os.path.join(args.data_root, f'amazon-{args.dataset}')
    
    
    try:
        # --- MODIFIED: Load unified node and edge files ---
        nodes_df = pd.read_csv(os.path.join(data_dir, 'all_nodes.csv'))
        edges_df = pd.read_csv(os.path.join(data_dir, 'all_edges.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease run the data preparation script first.")
        return
    nodes_df['rating'] = nodes_df['event'].apply(lambda x: x.split(":")[-1])


    # --- Generate or Load Embeddings ---
    print("\nStep 2: Generating or loading sentence embeddings...")
    # sbert_model_path = os.path.join(script_dir, '..', '..', 'sentence_transformers-all-MiniLM-L6-v2_safetensors')
    # sbert_model_path = os.path.abspath(sbert_model_path) 
    sbert_model_path = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # SBERT model is loaded lazily inside process_and_embed_user_events — only on a cache MISS.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x, mask, y, uid_map = process_and_embed_user_events(nodes_df, lambda: SentenceTransformer(sbert_model_path), args.max_events, data_dir)
    ratings = prepare_user_sequences(nodes_df, uid_map, args.max_events)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, mask, y = torch.from_numpy(x).float().to(device), torch.from_numpy(mask).bool().to(device), torch.from_numpy(y).long().to(device)
    ratings = torch.from_numpy(ratings).long().to(device)
    print("\nStep 3: Creating a single PyTorch Geometric data object...")
    uid_map_series = pd.Series(uid_map)
    data = build_graph_data(edges_df, uid_map_series, x, ratings, y, mask, args)
    
    print("\nStep 4: Loading canonical user split (shared across all fraud methods)...")
    from fraud_splits import get_canonical_user_split, masks_for_uids
    uid_split = get_canonical_user_split(data_dir, args.dataset, args.seed)
    train_mask, val_mask, test_mask = masks_for_uids(uid_split, uid_map, data.num_nodes)
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)
    
    print(f"Data splits: Training nodes: {data.train_mask.sum()}, Validation nodes: {data.val_mask.sum()}, Test nodes: {data.test_mask.sum()}")

    data = data.to(device)

    train_loader = NeighborLoader(data, 
                                input_nodes=data.train_mask,
                                num_neighbors=[args.num_neighbors] * len(args.gnn_layer_types.split(",")),
                                batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers,pin_memory=False,      )

    # --- ADD THIS BLOCK ---
    val_loader = NeighborLoader(data,
                                input_nodes=data.val_mask,
                                num_neighbors=[args.num_neighbors] * len(args.gnn_layer_types.split(",")),
                                batch_size=args.batch_size, # You can use a larger batch size for eval
                                shuffle=False, # No need to shuffle for validation
                                num_workers=args.num_workers,pin_memory=False)

    test_loader = NeighborLoader(data,
                                 input_nodes=data.test_mask,
                                 num_neighbors=[args.num_neighbors] * len(args.gnn_layer_types.split(",")),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,pin_memory=False)
    # --- END OF BLOCK ---

    # --- Handle Class Imbalance ---
    train_labels = data.y[data.train_mask]
    num_class_0 = (train_labels == 0).sum().item()
    num_class_1 = (train_labels == 1).sum().item()
    class_weights = torch.tensor([1.0, num_class_0 / num_class_1], dtype=torch.float).to(device)
    print(f"\nCalculated class weights for loss function: {class_weights}")

    sbert_embedding_dim = data.num_node_features  # = SBERT dim; avoids needing the model on a cache hit

    if args.seq_only:
        model = SentenceClassifier(in_channels=data.num_node_features, out_channels=2, args=args).to(device)
    else:
        if args.bridge:
            # Bridge: Transformer over event sequences, then GraphModel over users
            model = Bridge(
                in_channels=data.num_node_features,         # token feature dim (SBERT)
                args=args,
                gnn_hidden_channels=sbert_embedding_dim,    # as you used before
                dropout=args.dropout,
            ).to(device)
        else:
            # Original GraphModel path
            model = GraphModel(
                in_channels=data.num_node_features,
                hidden_channels=sbert_embedding_dim,
                gnn_layer_types=args.gnn_layer_types,
                dropout=args.dropout,
                args=args
            ).to(device)

        
    loss_fn = torch.nn.NLLLoss(weight=class_weights)
    t_train = time.time()

    # Supervised-frozen control (variant ii): warm up the sequence encoder on the task WITHOUT the
    # graph, then freeze it before training the graph module. Isolates joint optimization (variant
    # iii, end-to-end) from the supervised objective. Bridge-only; the end-to-end path is unchanged.
    freeze_seq = bool(getattr(args, 'freeze_encoder', False)) and args.bridge and not args.seq_only
    if freeze_seq:
        warm_opt = torch.optim.Adam(
            list(model.seq.parameters()) + list(model.warmup_head.parameters()),
            lr=args.lr, weight_decay=5e-4)
        print(f"\n[supervised-frozen] Phase A: warming up the sequence encoder for {args.encoder_warmup_epochs} epochs (no graph)...")
        for w in range(1, args.encoder_warmup_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                warm_opt.zero_grad()
                out = model(batch.x, batch.edge_index, pad_mask=batch.mask, ratings=batch.ratings, training=True, mode='warmup')
                bs = batch.batch_size
                loss = loss_fn(out[:bs], batch.y[:bs])   # seed users only; encoder never sees neighbor labels
                loss.backward()
                warm_opt.step()
        model.freeze_encoder()
        print("[supervised-frozen] Phase B: encoder frozen; training the graph module only.")

    # All params for end-to-end; only graph-module params if the encoder was frozen above.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    best_val_pr_auc = 0
    patience_counter = 0
    best_test_metrics = None
    best_pred_arrays = None
    # --- Training Loop ---
    print("\nStep 4: Starting GNN training...")
    for epoch in range(1, args.epochs + 1):
        # loss, train_acc, train_auc = train(model, train_data, train_mask,optimizer, loss_fn)
        # val_acc, val_roc_auc, _, _, val_loss = test(model, val_data, val_mask, loss_fn)
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ROC AUC: {val_roc_auc:.4f}')
        
        total_loss = 0
        total_acc = 0
        total_roc_auc = 0
        # breakpoint()
        loader = train_loader if not args.seq_only else [Data(x=data.x[data.train_mask], edge_index=data.edge_index, ratings = data.ratings[data.train_mask], y=data.y[data.train_mask], mask=data.mask[data.train_mask])]
        # --- NEW: Iterate over the NeighborLoader ---
        # for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        for batch in loader:
            loss, train_acc, train_roc_auc = train(model, batch, optimizer, loss_fn, freeze_seq=freeze_seq)
            total_loss += loss
            total_acc += train_acc
            total_roc_auc += train_roc_auc
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        avg_roc_auc = total_roc_auc / len(train_loader)

        val_metrics, _ = test(model, val_loader, device, loss_fn)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Train ROC AUC: {avg_roc_auc:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['max_f1']:.4f}, Val PR AUC: {val_metrics['pr_auc']:.4f}")
        
        # --- MODIFIED: Use the generic logger ---
        if logger:
            log_dict = {'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_acc, 'train_roc_auc': avg_roc_auc}
            for key, value in val_metrics.items():
                log_dict[f'val_{key}'] = value
            logger.log(log_dict)
            

        # --- Early Stopping Logic ---
        if val_metrics['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_metrics['pr_auc']
            patience_counter = 0
            # Save test metrics when validation is best
            test_metrics_current, _, _bp, _bl, _bn = test(model, test_loader, device, loss_fn, return_preds=True)
            best_test_metrics = test_metrics_current.copy()
            if args.dump_preds:
                best_pred_arrays = (_bn, _bp, _bl)   # (node ids, P(fraud), labels) at the best-val checkpoint
            # Optional: Save best model checkpoint here
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    train_wall_sec = time.time() - t_train

    # --- Final Evaluation ---
    print("\nStep 6: Final evaluation on the validation set...")
    val_metrics, val_pred = test(model,val_loader, device, loss_fn)
    
    print("\n--- Val Set Performance ---")
    for key, value in val_metrics.items():
        print(f"  - {key}: {value:.4f}")
    
    
    if logger:
        log_dict = {'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_acc, 'train_roc_auc': avg_roc_auc}
        for key, value in val_metrics.items():
            log_dict[f'val_{key}'] = value
        logger.log(log_dict)
        

    
    test_metrics,_ = test(model, test_loader, device, loss_fn)
    log_dict = {'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_acc, 'train_roc_auc': avg_roc_auc}
    for key, value in test_metrics.items():
        log_dict[f'test_{key}'] = value
    # Also log the best test metrics (when validation was best)
    if best_test_metrics:
        for key, value in best_test_metrics.items():
            log_dict[f'best_test_{key}'] = value
    log_dict['train_wall_sec'] = train_wall_sec

    if logger:
        logger.log(log_dict)
        logger.finish()

    # Serialize per-user (uid, P(fraud), label) at the best-val checkpoint for paired significance tests.
    if args.dump_preds and best_pred_arrays is not None:
        nids, probas, labels = best_pred_arrays
        idx2uid = {v: k for k, v in uid_map.items()}
        uids = np.array([idx2uid[int(i)] for i in nids])
        os.makedirs(args.preds_dir, exist_ok=True)
        _pp = os.path.join(args.preds_dir, f"{run_name}.npz")
        np.savez(_pp, uid=uids, score=probas, label=labels)
        print(f"[dump] wrote {_pp}  (n={len(uids)}, n_pos={int(labels.sum())})")

    print("\nClassification Report (Validation Set):")
    print(classification_report(data.y[data.val_mask].cpu(), val_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GNN for Fraud Detection")
    # --- Keep your existing arguments ---
    parser.add_argument("--dataset", type=str, choices=['musical', 'movies', 'clothing', 'electronics', 'home-kitchen', 'office-products'], default='musical')
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"),
                        help="dir with amazon-<dataset>/ (all_nodes.csv, all_edges.csv, splits/)")
    parser.add_argument("--log_dir", type=str, default="logs_delta",
                        help="root dir for LocalLogger JSONs; use a fresh dir for the canonical rerun")
    parser.add_argument("--suffix", type=str, default='rating', help="Suffix to wandb project.")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs.")
    parser.add_argument('--max_events', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for mini-batch training.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the data loader.")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early stopping.")
    
    # --- Add New Arguments for GraphModel ---
    parser.add_argument('--gnn_layer_types', type=str, default='TokenXAT', help='Comma-separated list of GNN layers to use, e.g., "TokenXAT,GCN"')
    # Bridge arguments
    # --- Transformer (sequence) hyperparameters ---
    parser.add_argument('--bridge', action='store_true', default=False,
                        help='If set, use Bridge: Transformer-over-sequence then GraphModel.')
    parser.add_argument('--tr_layers', type=int, default=2, help='Number of Transformer encoder layers.')
    parser.add_argument('--tr_nhead', type=int, default=4, help='Number of attention heads for Transformer.')
    parser.add_argument('--tr_ffn_dim', type=int, default=512, help='Transformer feedforward dimension.')
    parser.add_argument('--tr_dropout', type=float, default=0.1, help='Transformer dropout.')
    parser.add_argument('--tr_no_pos_enc', action='store_true', default=False,
                        help='Disable sinusoidal positional encoding if set.')
    parser.add_argument('--tr_out_dim', type=int, default=0,
                        help='Optional projection dimension before Transformer (0 = keep SBERT dim).')

    # Data split sizes
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1,
                        help="Comma-separated list of GNN layer types, e.g., 'TokenXAT,GCN,GCN'.")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads for GAT or TokenXAT')
    parser.add_argument('--compress_dim', type=int, default=4, help='Number of aggregation queries for TokenXAT')
    parser.add_argument('--aggregation_mode', type=str, default='gcn', choices=['gat', 'gcn'], help='Message passing mode for TokenXAT')
    parser.add_argument('--lr', type=float, default=1e-4)
    # --- Add other boolean/config flags used by TokenCrossAttnConv ---
    # These are just examples based on your class definition; adjust as needed.
    parser.add_argument('--pre_agg_proj', action='store_true', default=False)
    parser.add_argument('--post_agg_proj', action='store_true', default=False)
    parser.add_argument('--compress_mode', type=str, default='fixed', choices=['fixed', 'learned','mlp'])
    parser.add_argument('--add_self_loops', action='store_true', default=False)
    parser.add_argument('--qk_norm', action='store_true', default=False)
    parser.add_argument('--compress_first', action='store_true', default=False)
    parser.add_argument('--layer_norm', action='store_true', default=False)
    parser.add_argument('--seq_only', action='store_true', default=False)
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Supervised-frozen control (Bridge only): warm up the sequence encoder on the '
                             'task, then freeze it before training the graph module. Isolates joint '
                             'optimization from the supervised objective.')
    parser.add_argument('--encoder_warmup_epochs', type=int, default=50,
                        help='Epochs to supervise-train the sequence encoder before freezing (with --freeze_encoder).')
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--clf_layers', type=int, default=2, help="Number of layers in the MLP classifier.")
    parser.add_argument('--num_neighbors', type=int, default=10, help="Number of neighbors.")
    parser.add_argument('--use_wandb', action='store_true', help="Enable logging to Weights & Biases.") # <-- Add W&B flag
    parser.add_argument('--seed',  type=int, default=10, help="seed")
    parser.add_argument('--dump_preds', action='store_true', default=False,
                        help="Serialize per-user (uid, P(fraud), label) at the best-val checkpoint to --preds_dir "
                             "for paired significance tests (paired_bootstrap_prauc.py).")
    parser.add_argument('--preds_dir', type=str, default='preds', help="Output dir for --dump_preds npz files.")
    parser.add_argument('--rating_emb_dim', type=int, default=64, help="Dimension for the product rating embeddings.")
    parser.add_argument('--force_new_splits', action='store_true', default=False, 
                        help="Force creation of new data splits even if existing ones are found")


    args = parser.parse_args()
    main(args)
   