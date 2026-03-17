# GSoC 2026 — End-to-End Particle Collision Track Reconstruction
**Task 2i | ML4SCI × CMS Experiment @ CERN**

---

## Overview

This project addresses the GSoC 2026 test task for the **End-to-End (E2E) Deep Learning** project within the CMS experiment at CERN. The goal is to classify **quark jets vs. gluon jets** from point-cloud detector data using two Graph Neural Network (GNN) architectures.

The techniques demonstrated here — physics-informed graph construction using the ΔR metric, message passing, and dynamic edge features — are the foundational primitives for the actual summer project goal: **end-to-end track reconstruction** from low-level detector hits.

---

## Repository Structure

```
gsoc2026_e2e_tracking/
│
├── data/                   # Place the 3 .snappy.parquet files here
│
├── checkpoints/            # Saved best model weights (.pth files)
├── plots/                  # ROC curves and evaluation plots
│
├── graph_utils.py          # Physics-informed graph construction (ΔR metric)
├── dataset.py              # PyTorch Geometric custom Dataset class
├── models.py               # EdgeConv (DGCNN) and GATv2 architectures
├── train.py                # Training loop with checkpointing
├── evaluate.py             # ROC curves, AUC, background rejection
├── solution.ipynb          # Master notebook: full pipeline + discussion
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Dataset

Download the three `.snappy.parquet` files from CERNBox and place them in the `data/` directory:

```
https://cernbox.cern.ch/s/oolDBdQegsITFcv
```

**Files:**
- `QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet`  (~36k jets)
- `QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet`  (~47k jets)
- `QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet`  (~55k jets)

**Total: 139,306 jets | Perfectly balanced: 50% quark, 50% gluon**

**Data Schema:**
| Column   | Description |
|----------|-------------|
| `X_jets` | Nested list of shape `(3, 125, 125)` — 3 detector channels (channels-first) |
| `pt`     | Jet transverse momentum (GeV), range: 70.4 – 337.1 GeV |
| `m0`     | Jet mass (GeV) |
| `y`      | Binary label: `0` = quark jet, `1` = gluon jet |

**The 3 detector channels:**
| Index | Channel     | Value range      | Sparsity |
|-------|-------------|-----------------|---------|
| 0     | Track pT    | 0.000 – 10.031  | 99.7%   |
| 1     | ECAL energy | -0.217 – 12.672 | 97.8%   |
| 2     | HCAL energy | -0.001 – 0.133  | 97.8%   |

> **Note:** Data is stored as nested Python lists with one row group per jet.
> Always use PyArrow row-group reading — never `pd.read_parquet` on the full
> file, which causes out-of-memory errors.

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyG dependencies (adjust CUDA version as needed)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

---

## Running the Project

### Option A — Full Jupyter Notebook (Recommended)
```bash
jupyter lab solution.ipynb
```
Run all cells sequentially. The notebook covers data loading, graph construction,
training, evaluation, and discussion. Pre-built graphs are cached to `processed/`
after the first run — subsequent runs load instantly.

### Option B — Script Pipeline
```bash
# Step 1: Train both models
python train.py --model edgeconv --epochs 30
python train.py --model gatv2    --epochs 30

# Step 2: Evaluate and generate plots
python evaluate.py
```

---

## Results

| Model    | AUC-ROC | 1/εB @ εS=50% | Parameters |
|----------|---------|---------------|------------|
| EdgeConv | 0.7923  | 8.5           | 253,122    |
| GATv2    | 0.7715  | 7.6           | 275,138    |

Trained on 139,306 jets | Test set: 20,897 jets | Device: NVIDIA GPU (Kaggle T4)

---

## Key Physics Design Decisions

### Why ΔR instead of Euclidean distance?
The CMS detector uses a cylindrical geometry described by pseudorapidity (η)
and azimuthal angle (φ). The physically meaningful distance between two
particles is:

```
ΔR = sqrt(Δη² + Δφ²)
```

This metric is approximately Lorentz-invariant under longitudinal boosts,
unlike naive Euclidean distance in (x, y, z) space.

### φ Periodicity
The azimuthal angle φ wraps around at ±π. A naive subtraction `φ₁ - φ₂`
gives incorrect distances near the boundary. We correct for this:

```python
delta_phi = phi1 - phi2
delta_phi = delta_phi - 2*pi * round(delta_phi / (2*pi))
```

### Zero-Padding Removal
The dataset stores jets as padded arrays of shape `(3, 125, 125)` —
97–99% of pixels are zero. We reconstruct the true sparse point cloud
by keeping only active pixels (at least one non-zero channel value).
This typically yields 20–700 active particles per jet.

### Node Features
Each active pixel is mapped to physical coordinates and encoded as:
```
[log(1 + pT),  ECAL energy,  HCAL energy]
```
Log-scaling pT compresses its large dynamic range, stabilising training.

---

## Architectures

| Model | Key Idea | HEP Rationale |
|-------|----------|---------------|
| **EdgeConv (DGCNN)** | Edge features from neighbour differences `x_j - x_i` | Captures relative momentum flow in jet substructure |
| **GATv2** | Learned attention weights per edge | Focuses on high-pT core particles |

---

## Evaluation Metrics

Beyond standard accuracy, we use HEP-standard metrics:

- **ROC AUC**: Primary discrimination metric
- **Background Rejection** at 50% signal efficiency: `1/εB` at `εS = 0.5`
  - 1/εB = 8.5 means only 1 in 8.5 gluon jets pass when 50% of quark jets are accepted
  - Equivalently: 88.2% of background is rejected at the operating point

---

## Connection to Track Reconstruction

This test task demonstrates **graph-level classification** (quark vs. gluon).
The actual GSoC summer project — end-to-end track reconstruction — requires
**edge-level classification**: predicting whether two detector hits belong to
the same particle track. The same ΔR graph construction and message-passing
primitives apply directly. The key change is replacing `global_pool` with a
per-edge MLP prediction head.

---

## References

- CMS Collaboration. *End-to-end deep learning inference with CMSSW via ONNX
  using Docker*. CMS-DP-2023-036. [arXiv:2309.14254](https://arxiv.org/abs/2309.14254)
- Wang et al. *Dynamic Graph CNN for Learning on Point Clouds*. TOG 2019.
- Brody et al. *How Attentive are Graph Attention Networks?* ICLR 2022 (GATv2).
