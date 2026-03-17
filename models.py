"""
models.py
=========
Two Graph Neural Network architectures for Quark/Gluon jet classification.

Model A — EdgeConv (DGCNN)
  Dynamic Graph CNN. Each layer re-computes edge features from the
  difference between a node and its neighbours. Widely used in HEP jet
  physics due to its ability to capture local angular substructure.
  Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds",
             TOG 2019.

Model B — GATv2
  Graph Attention Network v2. Each edge is weighted by a learned attention
  score, allowing the model to focus on the most physically significant
  particles in a jet. More expressive than the original GAT.
  Reference: Brody et al., "How Attentive are Graph Attention Networks?",
             ICLR 2022.

Both models are designed to be lightweight (< 200k parameters) to
support eventual deployment in the CMS High-Level Trigger (HLT), where
inference latency is a hard constraint. ONNX export discussion is
provided in solution.ipynb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    EdgeConv, GATv2Conv, global_mean_pool, global_max_pool
)


# ---------------------------------------------------------------------------
# Shared MLP builder
# ---------------------------------------------------------------------------
def mlp(channels: list, batch_norm: bool = True) -> nn.Sequential:
    """
    Build a simple Multi-Layer Perceptron.

    Parameters
    ----------
    channels   : list of int — layer widths, e.g. [16, 64, 128]
    batch_norm : bool        — whether to include BatchNorm after each layer

    Returns
    -------
    nn.Sequential
    """
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Model A: EdgeConv (DGCNN)
# ---------------------------------------------------------------------------
class EdgeConvNet(nn.Module):
    """
    Dynamic Graph CNN for jet classification.

    Architecture:
        Input (N, 5)
        → EdgeConv1: edge MLP([2*5, 64, 64])      → (N, 64)
        → EdgeConv2: edge MLP([2*64, 128, 128])    → (N, 128)
        → EdgeConv3: edge MLP([2*128, 256])        → (N, 256)
        → global_mean_pool + global_max_pool       → (B, 512)
        → FC MLP(512 → 256 → 64 → 2)
        → log_softmax

    EdgeConv computes, for each edge (i, j):
        h_ij = MLP( [x_i, x_j - x_i] )
    Then aggregates: h_i = max_j h_ij

    The difference term (x_j - x_i) encodes the *relative* feature of
    neighbour j with respect to node i, capturing local substructure.

    Parameters
    ----------
    in_channels : int  — number of node features (default 3)
    dropout     : float — dropout probability in classifier head
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()

        # EdgeConv layers — edge MLP takes concatenated [x_i, x_j - x_i]
        # so input dimension is 2 * in_channels
        self.conv1 = EdgeConv(
            nn=mlp([2 * in_channels, 64, 64]),
            aggr='max'
        )
        self.conv2 = EdgeConv(
            nn=mlp([2 * 64, 128, 128]),
            aggr='max'
        )
        self.conv3 = EdgeConv(
            nn=mlp([2 * 128, 256]),
            aggr='max'
        )

        # Classifier head
        # After concat of mean-pool and max-pool: 256*2 = 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, batch):
        """
        Parameters
        ----------
        x          : Tensor (N, in_channels)  — node features
        edge_index : Tensor (2, E)            — graph connectivity
        batch      : Tensor (N,)              — batch vector

        Returns
        -------
        Tensor (B, 2)  — log-probabilities for [quark, gluon]
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        # Concatenate mean- and max-pooling for richer graph representation
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)

        return F.log_softmax(self.classifier(x), dim=1)

    def get_probabilities(self, x, edge_index, batch):
        """Return class probabilities (softmax) for evaluation."""
        return torch.exp(self.forward(x, edge_index, batch))


# ---------------------------------------------------------------------------
# Model B: GATv2
# ---------------------------------------------------------------------------
class GATv2Net(nn.Module):
    """
    Graph Attention Network v2 for jet classification.

    Architecture:
        Input (N, 5)
        → Linear projection → (N, 64)
        → GATv2Conv(64 → 64, heads=4, concat=True)  → (N, 256)
        → GATv2Conv(256 → 64, heads=4, concat=True)  → (N, 256)
        → GATv2Conv(256 → 128, heads=1, concat=False) → (N, 128)
        → global_mean_pool + global_max_pool          → (B, 256)
        → FC MLP(256 → 128 → 64 → 2)
        → log_softmax

    GATv2 computes attention coefficients as:
        α_ij = softmax_j( a · LeakyReLU(W · [x_i || x_j]) )

    Unlike GAT v1, this is a strictly more expressive function that can
    attend to any pair, not just pairs where one node dominates.
    In jet physics, attention weights naturally highlight high-pT "core"
    particles that carry the most discriminating information.

    Parameters
    ----------
    in_channels : int   — number of node features (default 3)
    dropout     : float — dropout on attention weights and classifier
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # Input projection to a common embedding space
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # GATv2 layers
        self.gat1 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=4,
            concat=True,       # output: 64*4 = 256
            dropout=dropout,
            add_self_loops=True
        )
        self.bn1 = nn.BatchNorm1d(256)

        self.gat2 = GATv2Conv(
            in_channels=256,
            out_channels=64,
            heads=4,
            concat=True,       # output: 64*4 = 256
            dropout=dropout,
            add_self_loops=True
        )
        self.bn2 = nn.BatchNorm1d(256)

        self.gat3 = GATv2Conv(
            in_channels=256,
            out_channels=128,
            heads=1,
            concat=False,      # output: 128
            dropout=dropout,
            add_self_loops=True
        )
        self.bn3 = nn.BatchNorm1d(128)

        # Classifier head — after concat of mean+max: 128*2 = 256
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, batch):
        """
        Parameters
        ----------
        x          : Tensor (N, in_channels)
        edge_index : Tensor (2, E)
        batch      : Tensor (N,)

        Returns
        -------
        Tensor (B, 2) — log-probabilities
        """
        x = self.input_proj(x)

        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.bn3(self.gat3(x, edge_index)))

        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)

        return F.log_softmax(self.classifier(x), dim=1)

    def get_probabilities(self, x, edge_index, batch):
        """Return class probabilities (softmax) for evaluation."""
        return torch.exp(self.forward(x, edge_index, batch))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_model(name: str, in_channels: int = 3, dropout: float = 0.3) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name        : str   — 'edgeconv' or 'gatv2'
    in_channels : int   — number of node features
    dropout     : float — dropout probability

    Returns
    -------
    nn.Module
    """
    name = name.lower()
    if name == "edgeconv":
        return EdgeConvNet(in_channels=in_channels, dropout=dropout)
    elif name == "gatv2":
        return GATv2Net(in_channels=in_channels, dropout=dropout)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'edgeconv' or 'gatv2'.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
