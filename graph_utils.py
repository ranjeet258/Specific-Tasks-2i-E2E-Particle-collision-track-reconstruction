import numpy as np
import torch
from torch_geometric.data import Data

ETA_MAX = 0.4
PHI_MAX = np.pi

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))

def extract_point_cloud(x_jet):
    """
    Input: x_jet shape (125, 125, 3) — channels-last
           Channel 0: Track pT
           Channel 1: ECAL energy
           Channel 2: HCAL energy

    Returns coords (N,2) and features (N,3).
    Removes zero-padded pixels — only keeps active hits.
    """
    active = np.any(x_jet != 0, axis=-1)       # (125,125) bool mask
    rows, cols = np.where(active)

    if len(rows) == 0:
        return None, None

    # Map pixel (i,j) → physical (Δη, Δφ) relative to jet axis
    delta_eta      = (rows - 62.0) / 62.0 * ETA_MAX
    delta_phi_vals = (cols - 62.0) / 62.0 * PHI_MAX
    coords = np.stack([delta_eta, delta_phi_vals], axis=1)  # (N,2)

    pt_vals   = x_jet[rows, cols, 0]
    ecal_vals = x_jet[rows, cols, 1]
    hcal_vals = x_jet[rows, cols, 2]

    # log(1+pT) stabilises the large dynamic range of pT
    log_pt   = np.log1p(np.abs(pt_vals))
    features = np.stack([log_pt, ecal_vals, hcal_vals], axis=1)  # (N,3)

    return coords, features

def build_physics_graph(x_raw, label, k=8):
    """
    x_raw: nested Python list of shape (3, 125, 125) — straight from parquet
    label: int 0=quark 1=gluon
    k    : number of nearest neighbours by ΔR

    Returns PyG Data object or None if jet has no active hits.
    """
    # Convert nested list → numpy, then channels-first → channels-last
    x_jet = np.array(x_raw, dtype=np.float32).transpose(1, 2, 0)  # (125,125,3)

    coords, features = extract_point_cloud(x_jet)
    if coords is None or len(coords) < 2:
        return None

    n    = len(coords)
    k_eff = min(k, n - 1)

    eta = coords[:, 0]
    phi = coords[:, 1]

    # Pairwise ΔR — the standard HEP angular distance metric
    deta     = eta[:, None] - eta[None, :]
    dphi_mat = delta_phi(phi[:, None], phi[None, :])
    dr_mat   = np.sqrt(deta**2 + dphi_mat**2)
    np.fill_diagonal(dr_mat, np.inf)

    # kNN edges by ΔR
    knn_idx = np.argsort(dr_mat, axis=1)[:, :k_eff]
    src = np.repeat(np.arange(n), k_eff)
    tgt = knn_idx.flatten()

    # Undirected: add reverse edges, remove duplicates
    row = np.concatenate([src, tgt])
    col = np.concatenate([tgt, src])
    edges = np.unique(np.stack([row, col], axis=1), axis=0)
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    return Data(
        x          = torch.tensor(features,  dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor([int(label)], dtype=torch.long)
    )
