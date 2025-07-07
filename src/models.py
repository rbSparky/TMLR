"""Defines the neural network architectures used in the project."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.cuda.amp import autocast
from . import config

class Projection(torch.nn.Module):
    """A simple linear projection layer with a ReLU activation."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """Projects the input tensor."""
        return F.relu(self.lin(x))

class MaskNet(torch.nn.Module):
    """
    Generates an edge mask 's' based on the features of incident nodes.

    The mask is computed for pairs of node features, passed through two linear
    layers, and finally a sigmoid activation.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x_proj, edge_index, chunk_size: int = None, use_amp_fwd: bool = False):
        """
        Computes edge mask scores in chunks to manage memory usage.

        Args:
            x_proj: Projected node features [num_nodes, proj_dim].
            edge_index: Edge index tensor [2, num_edges].
            chunk_size (int, optional): The number of edges to process at once.
            use_amp_fwd (bool): Whether to use AMP for this forward pass.

        Returns:
            A tensor of edge mask scores 's' of shape [num_edges].
        """
        effective_chunk_size = chunk_size if chunk_size is not None else config.MASK_CHUNK_SIZE
        num_edges = edge_index.size(1)
        s_list = []

        for i in range(0, num_edges, effective_chunk_size):
            edge_index_chunk = edge_index[:, i: i + effective_chunk_size]
            row, col = edge_index_chunk

            if edge_index_chunk.numel() == 0:
                continue

            with autocast(enabled=(use_amp_fwd and x_proj.is_cuda)):
                x_row = x_proj[row]
                x_col = x_proj[col]
                h_chunk = torch.cat([x_row, x_col], dim=1)
                h_chunk = F.relu(self.lin1(h_chunk))
                s_chunk = torch.sigmoid(self.lin2(h_chunk)).squeeze(-1)
            s_list.append(s_chunk)

        if not s_list:
            return torch.tensor([], device=x_proj.device)
        s = torch.cat(s_list, dim=0)
        return s

class TaskNet(torch.nn.Module):
    """
    The main GNN model for node classification, using GATConv as backbone.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, heads: int = 8, gat_dropout: float = 0.6):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        current_in_dim = in_dim

        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            concat_heads = not is_last_layer
            self.convs.append(GATConv(
                current_in_dim,
                hidden_dim,
                heads=heads,
                dropout=gat_dropout,
                edge_dim=1,
                concat=concat_heads
            ))
            current_in_dim = hidden_dim * heads if concat_heads else hidden_dim

        self.lin = torch.nn.Linear(current_in_dim, num_classes)
        self.dropout_layer = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Performs the forward pass for node classification.

        Args:
            x: Node features [num_nodes, num_features].
            edge_index: Edge index [2, num_edges].
            edge_weight (optional): Edge weights/mask [num_edges].
        
        Returns:
            Logits for each node [num_nodes, num_classes].
        """
        edge_attr = edge_weight
        if edge_attr is not None and edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = self.dropout_layer(x)

        x = self.lin(x)
        return x