# ---------------------------------------------------------------------------

# File: token_cross_attn_conv.py

# ---------------------------------------------------------------------------
import unittest
import math
import typing
from typing import Optional, Tuple, Union
import os 
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter, ModuleList, MultiheadAttention

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_batch, degree
from torch_geometric.nn import aggr
from torch_scatter import scatter, scatter_softmax # <-- Make sure to add this import at the top of your file

# import xformers.ops as xo
    
from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import spmm
# --- only needed for the scalar soft-max on a sparse index ---------------
from torch_geometric.utils import softmax
from torch.nn.attention import SDPBackend, sdpa_kernel
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._dynamo.disable()
# torch._dynamo.config.optimize_ddp = True
# torch._dynamo.config.cache_size_limit = 1000
# # create_block_mask = torch.compile(create_block_mask)


# # compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

 



class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)



class AttentionCompressor(nn.Module):
    """
    Compresses a sequence of tokens into a fixed-size set of k vectors.
    Supports two modes:
    - 'full': Standard nn.MultiheadAttention block.
    - 'flash': Efficient F.scaled_dot_product_attention block.
    """
    def __init__(self, feature_dim: int, num_queries: int, num_heads: int,
                 dropout: float = 0.1, mode: str = 'full'):
        super().__init__()
        
        self.mode = mode
        self.k = num_queries
        self.heads = num_heads
        self.feature_dim = feature_dim
        self.dropout_p = dropout
        self.induced_queries = Parameter(torch.zeros(1, self.k, feature_dim))
        nn.init.xavier_uniform_(self.induced_queries)
        self.dropout = nn.Dropout(self.dropout_p)

        if self.mode in ['full', 'flash']:
            self.induced_queries = Parameter(torch.zeros(1, self.k, feature_dim))
            nn.init.xavier_uniform_(self.induced_queries)
            if self.mode == 'full':
                self.attention = nn.MultiheadAttention(
                    embed_dim=feature_dim, num_heads=num_heads,
                    dropout=dropout, batch_first=True
                )
        elif self.mode == 'fixed':
            # Determine aggregation methods based on k
            if self.k == 1:
                self.aggr_methods = ["mean"]
            elif self.k == 2:
                self.aggr_methods = ["mean", "max"]
            elif self.k == 3:
                self.aggr_methods = ["mean", "max", "min"]
            elif self.k >= 4:
                self.aggr_methods = ["mean", "max", "min", "last"]
                # Ensure self.k matches the number of aggregations
                self.k = 4
            else:
                raise ValueError(f"k must be >= 1 for fixed mode, but got {self.k}")
                # --- ADDED: MLP Mode Initialization ---
        elif self.mode == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim * self.k)
            )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, training = True) -> torch.Tensor:
        """Dispatches to the appropriate forward method based on the mode."""
        self.training = training
        # breakpoint()
        if self.mode == 'full':
            return self.forward_full(x, key_padding_mask)
        elif self.mode == 'flash':
            return self.forward_flash(x, key_padding_mask)
        elif self.mode == 'fixed':
            return self.forward_fixed(x, key_padding_mask)
        elif self.mode == 'mlp':
            return self.forward_mlp(x, key_padding_mask)

    def forward_full(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass using the standard nn.MultiheadAttention block."""
        batch_size = x.size(0)
        queries = self.induced_queries.expand(batch_size, -1, -1)
        
        
        output, _ = self.attention(
            query=queries, key=x, value=x,
            key_padding_mask=~key_padding_mask.bool() if key_padding_mask is not None else None
        )
        
        return output

    def forward_flash(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass using the optimized F.scaled_dot_product_attention.
        """
        B, S, D = x.shape
        batch_size = x.size(0)
        queries = self.induced_queries.expand(batch_size, -1, -1)
        
        

        # `key_padding_mask` usually has True for valid tokens.
        # The attention function needs True for tokens to MASK (ignore).
        final_attn_mask = ~key_padding_mask.bool() if key_padding_mask is not None else None

        queries_len = queries.size(1)  # Length of the query sequence
        final_attn_mask = None
        if key_padding_mask is not None:
            # 1. Create the 2D boolean mask with the correct convention
            mask_2d = ~key_padding_mask.bool()  # Shape: (B, S)
            
            # 2. Manually expand to the 3D shape (B, L, S)
            final_attn_mask = mask_2d.unsqueeze(1).expand(batch_size, queries_len, S)

        attn_output = F.scaled_dot_product_attention(
            queries,
            x,
            x,
            attn_mask=final_attn_mask,  # ✅ Pass the correctly shaped 3D mask
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        
        return attn_output
    
    def forward_fixed(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass using fixed pooling operations."""
        B, S, D = x.shape
        outputs = []

        # If no mask is provided, assume all tokens are valid.
        # The mask should have True for valid tokens for pooling logic.
        mask = key_padding_mask.bool() if key_padding_mask is not None else torch.ones(B, S, device=x.device, dtype=torch.bool)
        
        for aggr_type in self.aggr_methods:
            if aggr_type == "mean":
                masked_x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
                sum_x = masked_x.sum(dim=1)
                num_valid = mask.sum(dim=1, keepdim=True).float()
                outputs.append(sum_x / (num_valid))
            
            elif aggr_type == "max":
                masked_x = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
                max_pool, _ = masked_x.max(dim=1)
                outputs.append(max_pool)
            
            elif aggr_type == "min":
                masked_x = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).max)
                min_pool, _ = masked_x.min(dim=1)
                outputs.append(min_pool)

            elif aggr_type == "last":
                s_indices = torch.arange(S, device=x.device)
                last_valid_indices = (s_indices * mask.int()).argmax(dim=1)
                last_pool = x[torch.arange(B), last_valid_indices]
                outputs.append(last_pool)
        # breakpoint()
        
        return torch.stack(outputs, dim=1)
    
    # --- ADDED: MLP Forward Method ---
    def forward_mlp(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass using mean pooling followed by an MLP.
        """
        B, S, D = x.shape
        
        # 1. Masked Mean Pooling
        mask = key_padding_mask.bool() if key_padding_mask is not None else torch.ones(B, S, device=x.device, dtype=torch.bool)
        masked_x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        sum_x = masked_x.sum(dim=1)
        num_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        mean_pooled = sum_x / num_valid # Shape: (B, D)
        
        # 2. Pass through MLP
        mlp_output = self.mlp(mean_pooled) # Shape: (B, D * k)
        
        # 3. Reshape to (B, k, D)
        output = mlp_output.view(B, self.k, self.feature_dim)
        return output


def linear_attn(q, k, v, kv_mask=None):
  """
  Memory-linear attention from Katharopoulos et al., 2020.

  Shapes
  -------
  q, k, v : (B, H, N, D)
    B – batch (here := #edges)
    H – heads
    N – sequence length (T tokens)
    D – head dim
  kv_mask : (B, N) or None
    Boolean mask that marks valid key / value tokens.
  """
  dim = q.shape[-1]

  if kv_mask is not None:
    mask_value = torch.finfo(q.dtype).min
    # broadcast to (B, 1, N, 1)
    mask = kv_mask[:, None, :, None]
    k = k.masked_fill(~mask, mask_value)
    v = v.masked_fill(~mask, 0.)

  q = q.softmax(dim=-1)     # over D
  k = k.softmax(dim=-2)     # over N

  q = q * dim ** -0.5      # scale queries

  # Context: (B, H, D, D)  – \sum_n k_n v_nᵀ
  context = torch.einsum('bhnd,bhne->bhde', k, v)
  # Output : (B, H, N, D)  – q_n · context
  attn  = torch.einsum('bhnd,bhde->bhne', q, context)
  return attn





class TokenCrossAttnConv(MessagePassing):

    r"""
    **Token-level cross-attention GNN** (minimal patch over `GATConv`).

    *Input*: `x` has shape **(N, T, F_in)** – a length-`T` sequence per node. 
    For every directed edge *u → v* we compute a full Transformer attention
    block between the `T` tokens of `u` (*keys/values*) and the `T` tokens of
    `v` (*queries*), then aggregate neighbour messages **token-wise**.
    The output keeps the same shape *(N, T, H·F_out)* (or averaged if
    `concat=False`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        # --- New arguments for flexible aggregation ---
        pre_agg_proj: bool = False,
        post_agg_proj: bool = False,
        # aggregations: list[str] = ['mean', 'max', 'min'], # Used only if mode is 'fixed'
        aggregations: list[str] = ['mean'],
        compress_mode: str = 'fixed',
        num_agg_queries: int = 4, # Used only if mode is 'learned'
        use_qk_norm: bool = False,
        compress_first: bool = False,
        aggregation_mode: str = "gat",
        gcn_edge_weight: bool = False,
        use_layer_norm: bool = False,
        residual: bool = False,
        num_tokens: int = 10,
        # ---
        **kwargs,
        ):
        kwargs.setdefault("aggr", "add")
        # NOTE: `node_dim=0` because we keep (N, T, d) and treat dim0 as nodes
    
        # At the end of the method:
        

        super().__init__(node_dim=0, **kwargs)

        # self.aggregations = aggregations
        self.in_channels = in_channels     # F_in
        self.out_channels = out_channels    # F_out
        self.heads = heads
        self.concat = concat
        self.dropout_p = dropout
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.compress_mode = compress_mode
        self.use_qk_norm = use_qk_norm
        self.aggregation_mode = aggregation_mode
        self.num_tokens = num_tokens
        self.residual = residual
        # self.k = len(aggregations)
        ## MODIFIED: Set aggregation based on mode. GCN uses 'add'.
        # if self.gcn_edge_weight:
        if self.aggregation_mode == 'gcn' or self.aggregation_mode == 'gat':
            kwargs.setdefault("aggr", "add")
        else:
            kwargs.setdefault("aggr", "mean")


        self.compress_first = compress_first
        agg_feature_dim = self.heads * self.in_channels 
        

        
        if self.use_qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.in_channels),RMSNorm(self.in_channels)
        if self.aggregation_mode == 'gat':
            feature_dim = self.heads * self.in_channels
            # The input size is 2*feature_dim because we will concatenate
            # the central node's features and the neighbor's features.
            self.agg_attn_lin = Linear(2 * feature_dim, 1, bias=False)
            

        self.post_compression_norm = nn.LayerNorm(agg_feature_dim)
            

        # --- Unified Compressor Instantiation ---
        self.compressor = AttentionCompressor(
            feature_dim=agg_feature_dim,
            num_queries=num_agg_queries,
            num_heads=self.heads,
            dropout=self.dropout,
            mode= 'full' if compress_mode == 'learned' else compress_mode
        )
        # The true number of output vectors from the compressor
        self.k = self.compressor.k 
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.post_attention_norm = nn.LayerNorm(self.in_channels) # ✅ ADD THIS
            self.post_aggregation_norm = nn.LayerNorm(agg_feature_dim) # ✅ ADD THIS
        
        
        self.pre_agg_proj = pre_agg_proj
        self.post_agg_proj = post_agg_proj
        
        
        if self.aggregation_mode == 'gcn' or self.aggregation_mode == 'gat':
            self.lin = nn.Linear(self.in_channels, agg_feature_dim, bias=False)

        self.lin_all = Linear(self.in_channels, 3 * self.heads * self.in_channels, bias=False,weight_initializer='glorot')

        if pre_agg_proj:
            self.pre_aggregation_projection = nn.Linear(agg_feature_dim, agg_feature_dim)

        # Conditionally create the post-aggregation projection layer
        if post_agg_proj:
            self.post_aggregation_projection = nn.Linear(agg_feature_dim, agg_feature_dim)


        # breakpoint()
        if bias:
            self.bias = Parameter(torch.empty(self.k * agg_feature_dim))
        else:
            self.register_parameter('bias', None)
        total_out_channels = out_channels * (heads if concat else 1)
        # if residual:
        #     self.res = Linear(
        #         in_channels * self.k
        #         if isinstance(in_channels, int) else in_channels[1],
        #         total_out_channels * self.k,
        #         bias=False,
        #         weight_initializer='glorot',
        #     )
        # else:
        #     self.register_parameter('res', None)
        

        # The input dimension for the residual layer is the flattened
        # dimension of the original input tensor (N, T, F_in).
        input_dim_for_res = self.num_tokens * self.in_channels

        # The output dimension must match the final output of the main path.
        output_dim_for_res = self.heads * self.out_channels * self.k

        # This linear layer learns the best projection.
        # Setting bias=False is a common practice for residual projections.
        self.res = Linear(input_dim_for_res, output_dim_for_res, bias=False)


            
        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        # Reset all sub-modules
        if hasattr(self, 'lin'): self.lin.reset_parameters()
        if hasattr(self, 'lin_all'): self.lin_all.reset_parameters()
        if hasattr(self, 'agg_attn_lin'): self.agg_attn_lin.reset_parameters()
        if hasattr(self, 'pre_aggregation_projection'): self.pre_aggregation_projection.reset_parameters()
        if hasattr(self, 'post_aggregation_projection'): self.post_aggregation_projection.reset_parameters()
        if hasattr(self, 'bias'): nn.init.zeros_(self.bias)
        if self.res is not None:
            self.res.reset_parameters()

    def forward(self,
                x: Tensor,  # (N, T, F_in)
                edge_index: Adj,
                user_timestamps = None,
                mask: OptTensor = None, # MODIFIED: Accept an explicit mask
                size: Size = None,
                return_attention_weights: bool = False,
                training: bool = True
                ):
        assert x.dim() == 3, "`x` must be (N, T, F_in)"
        self.training = training
        # breakpoint()
        # 1. Apply the learnable skip connection to the ORIGINAL input `x`
        res = None
        H, C = self.heads, self.in_channels
        N, T, _ = x.shape
        # breakpoint()
        if self.residual:
            # Flatten the input for the linear residual layer
            res = self.res(x.view(N, -1))
            

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        propagate_mask = mask if not self.compress_first else None
        
        if self.compress_first:
            x_reshaped = x.view(N, T, H * x.shape[-1])
            x = self.compressor(x_reshaped,  key_padding_mask=mask, training = self.training)

            N, T, _ = x.shape
            if self.use_layer_norm:
                x = self.post_compression_norm(x)


        qkv = self.lin_all(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(N, T, H, C)
        K = K.view(N, T, H, C)
        V = V.view(N, T, H, C)
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
            
        # --- MODIFIED: Create distinct logic paths for each message mode ---
        out = None

        if self.aggregation_mode == 'gcn':
            # 1. Prepare for GCN-style message (simple linear transformation)
            # 2. Calculate GCN normalization weights
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            norm = norm.to(x.device)

        else:
            norm = torch.ones(edge_index.shape[1])
            norm = norm.to(x.device)
        out = self.propagate(edge_index, Q=Q, K=K, V=V, x=x, pad_mask=propagate_mask, user_timestamps=user_timestamps, size=size, edge_weight=norm)

        if res is not None:
            # breakpoint()
            out = out + res
        # breakpoint()
        if self.bias is not None:
            out = out + self.bias  # <-- This step is missing
        return out
        


    # ------------------------------------------------------------------ #
    def elu_feature_map(self, x: Tensor) -> Tensor:
        return F.elu(x) + 1.0   # always > 0

    # def propagate(self,edge_index, size = None, **kwargs):
    #     return super().propagate(edge_index, size = size, **kwargs)
    def message(
            self,
            Q_i: Optional[Tensor] = None,
            K_j: Optional[Tensor] = None,
            V_j: Optional[Tensor] = None, 
            user_timestamps_j: Tensor= None, 
            index: Tensor= None, 
            ptr: OptTensor= None, 
            size_i: Optional[int]= None, 
            training :bool = True, 
            edge_weight: OptTensor = None
        ):
        # 1. (E, T, H, C) → (E, H, T, C)
        q = Q_i.permute(0, 2, 1, 3)
        k = K_j.permute(0, 2, 1, 3)
        v = V_j.permute(0, 2, 1, 3)
        

        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,  # token-to-token is fully connected
            )           # ➜ (E, H, T, C)

        except Exception as e:
            print("Qurey: ", q)
            print("Key: ", k)
            print("Value: ", v)
            print(f"DEBUG: q.shape = {q.shape}") # Add this line
            print(e)
            exit()
            
        
        # breakpoint()
        if edge_weight is not None and self.aggregation_mode == 'gcn':
            # Reshape for broadcasting: (E,) -> (E, 1, 1, 1) to match out's shape
            out = edge_weight.view(-1, 1, 1, 1) * out
            
        
        out = out.permute(0, 2, 1, 3)
        # breakpoint()
        return out





    # Add this helper method to your class if it's not already there
    def _gat_self_attention_aggregate(self, neighbor_messages, central_node_features, 
                                    index, dim_size):
        """
        Performs GAT self-attention aggregation.
        Assumes neighbor_messages and central_node_features are in the same space.
        """
        # a. Create summary vectors for both central nodes and neighbors.
        summary_i = central_node_features.mean(dim=1)
        summary_j = neighbor_messages.mean(dim=1)

        # b. Concatenate features for the attention mechanism.
        attn_input = torch.cat([summary_i, summary_j], dim=-1)

        # c. Calculate and normalize attention scores (alpha).
        attn_logits = F.leaky_relu(self.agg_attn_lin(attn_input))
        alpha = scatter_softmax(attn_logits, index, dim=0)

        # d. Apply weights and aggregate.
        weighted_messages = neighbor_messages * alpha.view(-1, 1, 1)
        return scatter(weighted_messages, index, dim=0, dim_size=dim_size, reduce='sum')


    # Replace your old aggregate function with this new, organized one
    def aggregate(self, inputs: Tensor, index: Tensor, x: Tensor, edge_index: Adj,
                pad_mask_i: OptTensor, dim_size: Optional[int] = None) -> Tensor:
        """
        Performs a two-stage aggregation:
        1. Prepares central node and neighbor features based on the compress_first flag.
        2. Aggregates the prepared features based on the compress_mode.
        """
        E, T, H, C = inputs.shape
        feature_dim = H * C

        # --- Part 1: Feature Preparation ---
        # The goal of this block is to produce `central_node_feats` and `neighbor_feats`,
        # ensuring both are in the correct (potentially compressed) feature space.

        central_node_feats, neighbor_feats = None, None
        row, col = edge_index

        # Reshape neighbor messages from (E, T, H, C) to (E, T, D)
        processed_neighbor_messages = inputs.contiguous().view(E, T, feature_dim)
        if self.use_layer_norm:
            processed_neighbor_messages = self.post_attention_norm(processed_neighbor_messages)

        if self.compress_first:
            # Workflow: COMPRESS-FIRST
            # `x` is already compressed in forward(). Messages (`inputs`) are from compressed `x`.
            central_node_feats = x[row]
            neighbor_feats = processed_neighbor_messages
        else:
            # Workflow: COMPRESS-DURING-AGGREGATION
            # Apply pre-projection before compressing the neighbor messages.
            if self.pre_agg_proj:
                processed_neighbor_messages = self.pre_aggregation_projection(processed_neighbor_messages)
            
            # We must compress both central node features and neighbor messages here.
            central_node_feats = self.compressor(x[row], training=self.training)
            neighbor_feats = self.compressor(processed_neighbor_messages, key_padding_mask=pad_mask_i, training=self.training)

        # --- Part 2: Final Aggregation ---
        # Now, we use the prepared features to perform the chosen aggregation.
        final_output = None
        if self.aggregation_mode == 'gat':
            final_output = self._gat_self_attention_aggregate(
                neighbor_messages=neighbor_feats,
                central_node_features=central_node_feats,
                index=index,
                dim_size=dim_size
            )
        else:  # 'mean' or 'gcn'
            reduce_op = 'sum' if self.aggregation_mode in ['gcn','gat'] else 'mean'
            final_output = self._scatter_aggregate(neighbor_feats, index, dim_size)

        # --- Part 3: Post-Processing ---
        if self.use_layer_norm:
            final_output = self.post_aggregation_norm(final_output)

        if self.post_agg_proj:
            final_output = F.relu(self.post_aggregation_projection(final_output))

        return final_output.view(dim_size, -1) if self.concat else final_output.mean(dim=1)


    def _scatter_mean_aggregate(self, inputs: Tensor, index: Tensor, dim_size: int) -> Tensor:
        """Generic helper to aggregate neighbor messages using scatter-mean."""
        num_items, seq_len, feature_dim = inputs.shape
        inputs_flat = inputs.reshape(num_items, seq_len * feature_dim)
        
        aggregated_flat = scatter(inputs_flat, index, dim=0, dim_size=dim_size, reduce='sum' if (self.aggregation_mode == 'gcn') else 'mean')
        
        return aggregated_flat.view(dim_size, seq_len, feature_dim)
    
    def _scatter_aggregate(self, inputs: Tensor, index: Tensor, dim_size: int) -> Tensor:
        """
        A generic helper to aggregate neighbor messages using scatter.
        """
        num_messages, seq_len, feature_dim = inputs.shape
        inputs_flat = inputs.reshape(num_messages, seq_len * feature_dim)
        
        aggregated_flat = scatter(inputs_flat, index, dim=0, dim_size=dim_size,  reduce='sum' if (self.aggregation_mode == 'gcn') else 'mean')
        
        return aggregated_flat.view(dim_size, seq_len, feature_dim)




def main() -> None:
  torch.manual_seed(0)
  # ---- tiny dummy graph --------------------------------------------------
  N, T, d_in = 6, 12, 32    # nodes, tokens, input-dim
  d_out, H  = 16, 4      # output per head, #heads
  edge_index = torch.tensor([[0, 0, 1, 2, 3, 4],
                [1, 2, 2, 3, 4, 5]])  # (2, E)
  x = torch.randn(N, T, d_in)

  # ---- instantiate the layer --------------------------------------------

  layer = TokenCrossLinAttnConv(
    in_channels = d_in,
    out_channels = d_out,
    heads    = H,     # concat=True by default
    dropout   = 0.0,
  )

  # ---- 1. forward --------------------------------------------------------
  out = layer(x, edge_index)      # (N, T, H*d_out)
  print("output shape :", out.shape)   # expect (6, 12, 64)
  print("contains NaN :", torch.isnan(out).any().item())


  assert out.shape == (N, T, H * d_out), "Shape mismatch!"
  # ---- 2. backward -------------------------------------------------------
  loss = out.pow(2).mean()        # dummy objective
  loss.backward()

  nz_grad = sum((p.grad is not None and p.grad.abs().sum() > 0)
         for p in layer.parameters() if p.requires_grad)

  print(f"non-zero grads for {nz_grad}/{sum(p.requires_grad for p in layer.parameters())} params")
  assert nz_grad > 0, "No gradients flowing!"

  # ---- 3. optional attention weights ------------------------------------
  # out2, (ei, alpha) = layer(x, edge_index, return_attention_weights=True)
  out2 = layer(x, edge_index)
  # print("alpha shape  :", alpha.shape) # (E, T, T) averaged over heads
  print("\nAll basic checks passed!")

# if __name__ == "__main__":

#   main()
  
  
  
# --- Test Case ---
class TestAggregationMethods(unittest.TestCase):
    def setUp(self):
        """Set up all necessary parameters and tensors for the tests."""
        self.N, self.T, self.d_in = 6, 12, 32
        self.d_out, self.H = 16, 4
        self.aggregations = ['mean', 'max', 'min']
        
        self.layer = TokenCrossAttnConv(
            in_channels=self.d_in,
            out_channels=self.d_out,
            heads=self.H,
            aggregations=self.aggregations
        )
        
        self.x = torch.randn(self.N, self.T, self.d_in)
        self.edge_index = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 2, 3, 4, 5]])
        
        self.pad_mask = torch.ones(self.N, self.T, dtype=torch.bool)
        for i in range(self.N):
            num_masked = torch.randint(0, self.T // 2, (1,)).item()
            if num_masked > 0:
                self.pad_mask[i, -num_masked:] = False

    def test_outputs_are_identical(self):
        """The main test: run both methods and assert their outputs are the same."""
        print("\nRunning test: test_outputs_are_identical...")
        
        # Run with the first aggregation method
        output1 = self.layer(self.x, self.edge_index, mask=self.pad_mask, aggregation_method='flatten_and_scatter')
        print(f"Output 1 (flatten) shape: {output1.shape}")

        # Run with the second aggregation method
        output2 = self.layer(self.x, self.edge_index, mask=self.pad_mask, aggregation_method='two_stage')
        print(f"Output 2 (two-stage) shape: {output2.shape}")

        self.assertTrue(
            torch.allclose(output1, output2, atol=1e-5),
            "The outputs of the two aggregation functions are not identical."
        )
        print("✅ Test Passed: Outputs are identical.")

def demonstration_main():
  """Original main function to demonstrate layer functionality."""
  torch.manual_seed(0)
  N, T, d_in = 6, 12, 32
  d_out, H  = 16, 4
  edge_index = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 2, 3, 4, 5]])
  x = torch.randn(N, T, d_in)

  layer = TokenCrossAttnConv(
    in_channels = d_in,
    out_channels = d_out,
    heads    = H,
    dropout   = 0.0,
  )

  out = layer(x, edge_index)
  print("\n--- Demonstration ---")
  print("output shape :", out.shape)
  print("contains NaN :", torch.isnan(out).any().item())
  assert out.shape == (N, T, H * d_out), "Shape mismatch!"
  
  loss = out.pow(2).mean()
  loss.backward()
  nz_grad = sum((p.grad is not None and p.grad.abs().sum() > 0) for p in layer.parameters() if p.requires_grad)
  print(f"non-zero grads for {nz_grad}/{sum(p.requires_grad for p in layer.parameters())} params")
  assert nz_grad > 0, "No gradients flowing!"
  print("All basic checks passed!")


if __name__ == "__main__":
    # Run the unit test suite by default
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    