"""
evaluate.py
===========
CERN-standard evaluation for the Quark/Gluon jet GNN classifiers.

Computes:
  - ROC curves (both models on one plot)
  - AUC-ROC scores
  - Background rejection factor (1/εB) at 50% signal efficiency
  - Inference speed in ms/batch

Usage:
    python evaluate.py
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import JetGraphDataset, split_dataset
from models import get_model, count_parameters


# ---------------------------------------------------------------------------
# Load a saved checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(model_name: str, root: str, device: torch.device,
                    in_channels: int = 3):
    """Load model weights from the best checkpoint."""
    ckpt_path = os.path.join(root, "checkpoints", f"best_{model_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            f"Run train.py --model {model_name} first."
        )
    model = get_model(model_name, in_channels=in_channels)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} checkpoint (epoch {ckpt['epoch']}, "
          f"val AUC={ckpt['val_auc']:.4f})")
    return model


# ---------------------------------------------------------------------------
# Collect predictions on the test set
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Run model on all batches and collect true labels and predicted probabilities.

    Returns
    -------
    labels : np.ndarray — true binary labels
    scores : np.ndarray — predicted probability for the gluon (positive) class
    """
    all_labels = []
    all_scores = []

    for batch in loader:
        batch = batch.to(device)
        probs = model.get_probabilities(batch.x, batch.edge_index, batch.batch)
        all_scores.append(probs[:, 1].cpu().numpy())   # gluon probability
        all_labels.append(batch.y.squeeze().cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_scores)


# ---------------------------------------------------------------------------
# Background rejection at fixed signal efficiency
# ---------------------------------------------------------------------------
def background_rejection(labels, scores, signal_eff: float = 0.5):
    """
    Compute the background rejection factor 1/εB at a given signal efficiency εS.

    In HEP jet tagging, the standard benchmark is:
        "At what rate do we accept background gluon jets
         when we accept 50% of quark jets?"

    A higher 1/εB is better — it means fewer gluon jets slip through
    for every quark jet we accept.

    Parameters
    ----------
    labels      : array-like — true binary labels (1 = signal = gluon)
    scores      : array-like — predicted probability for the gluon class
    signal_eff  : float      — target signal efficiency (default 0.5)

    Returns
    -------
    rejection : float  — 1/εB at the specified signal efficiency
    threshold : float  — score threshold that achieves signal_eff
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find the threshold closest to the target signal efficiency (TPR)
    idx = np.argmin(np.abs(tpr - signal_eff))
    eps_B = fpr[idx]          # background efficiency at this point
    threshold = thresholds[idx]

    if eps_B == 0:
        rejection = np.inf
    else:
        rejection = 1.0 / eps_B

    return rejection, threshold, fpr, tpr


# ---------------------------------------------------------------------------
# Inference speed benchmark
# ---------------------------------------------------------------------------
def measure_inference_speed(model, loader, device, n_warmup: int = 5):
    """
    Measure average forward-pass time in milliseconds per batch.

    Parameters
    ----------
    model    : nn.Module
    loader   : DataLoader
    device   : torch.device
    n_warmup : int — number of warm-up batches before timing starts

    Returns
    -------
    ms_per_batch : float
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(batch.x, batch.edge_index, batch.batch)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if i >= n_warmup:
                times.append(elapsed_ms)

            if i >= n_warmup + 50:   # time 50 batches
                break

    return float(np.mean(times))


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------
def plot_roc_curves(results: dict, save_path: str):
    """
    Plot ROC curves for both models on one figure — CMS style.

    Parameters
    ----------
    results   : dict — {model_name: {"fpr", "tpr", "auc", "rejection"}}
    save_path : str  — output path for the PNG
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"edgeconv": "#1f77b4", "gatv2": "#d62728"}
    labels_map = {"edgeconv": "EdgeConv (DGCNN)", "gatv2": "GATv2"}

    for name, res in results.items():
        ax.plot(res["fpr"], res["tpr"],
                color=colors.get(name, "gray"),
                lw=2,
                label=f"{labels_map.get(name, name)}  AUC={res['auc']:.4f}")

    # Mark the 50% signal efficiency operating point for each model
    for name, res in results.items():
        idx = np.argmin(np.abs(np.array(res["tpr"]) - 0.5))
        ax.scatter(res["fpr"][idx], res["tpr"][idx],
                   color=colors.get(name, "gray"),
                   marker="*", s=200, zorder=5)

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

    ax.set_xlabel("Background Efficiency (εB)", fontsize=13)
    ax.set_ylabel("Signal Efficiency (εS)", fontsize=13)
    ax.set_title("ROC Curve — Quark/Gluon Jet Tagger", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # CMS preliminary annotation
    ax.text(0.05, 0.97, "CMS Simulation Preliminary",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", style="italic",
            color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {save_path}")


# ---------------------------------------------------------------------------
# Training history plot
# ---------------------------------------------------------------------------
def plot_training_history(root: str, save_path: str):
    """Plot loss and AUC curves from saved training histories."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"edgeconv": "#1f77b4", "gatv2": "#d62728"}
    labels_map = {"edgeconv": "EdgeConv", "gatv2": "GATv2"}

    for name in ["edgeconv", "gatv2"]:
        hist_path = os.path.join(root, "results", f"history_{name}.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            history = json.load(f)

        epochs     = [h["epoch"]     for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss   = [h["val_loss"]   for h in history]
        val_auc    = [h["val_auc"]    for h in history]

        c = colors.get(name, "gray")
        lbl = labels_map.get(name, name)

        axes[0].plot(epochs, train_loss, color=c, lw=2, label=f"{lbl} train")
        axes[0].plot(epochs, val_loss,   color=c, lw=2, linestyle="--",
                     label=f"{lbl} val")
        axes[1].plot(epochs, val_auc,    color=c, lw=2, label=lbl)

    axes[0].set_title("Loss vs Epoch");     axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL Loss");          axes[0].legend()
    axes[1].set_title("Val AUC vs Epoch");  axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC");           axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(root: str = ".", batch_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load dataset and get test split
    dataset = JetGraphDataset(root=root)
    _, _, test_data = split_dataset(dataset)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    results = {}
    summary_rows = []

    for model_name in ["edgeconv", "gatv2"]:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*50}")

        try:
            model = load_checkpoint(model_name, root, device)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Predictions
        labels, scores = get_predictions(model, test_loader, device)

        # AUC
        auc = roc_auc_score(labels, scores)

        # Background rejection at 50% signal efficiency
        rejection, threshold, fpr, tpr = background_rejection(labels, scores, 0.5)

        # Inference speed
        ms_per_batch = measure_inference_speed(model, test_loader, device)
        n_params = count_parameters(model)

        print(f"  AUC-ROC:              {auc:.4f}")
        print(f"  Background Rejection: {rejection:.1f}  "
              f"(1/εB at εS=50%)")
        print(f"  Threshold @ εS=50%:   {threshold:.4f}")
        print(f"  Inference speed:      {ms_per_batch:.2f} ms/batch "
              f"(batch={batch_size})")
        print(f"  Parameters:           {n_params:,}")

        results[model_name] = {
            "auc": auc,
            "rejection_at_50": rejection,
            "threshold_at_50": threshold,
            "ms_per_batch": ms_per_batch,
            "n_params": n_params,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }

        summary_rows.append({
            "model": model_name,
            "auc": round(auc, 4),
            "rejection_at_50pct_sig_eff": round(rejection, 1),
            "ms_per_batch": round(ms_per_batch, 2),
            "n_params": n_params
        })

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    plots_dir = os.path.join(root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if results:
        plot_roc_curves(results,
                        os.path.join(plots_dir, "roc_comparison.png"))
        plot_training_history(root,
                              os.path.join(plots_dir, "training_history.png"))

    # ---------------------------------------------------------------
    # Save summary JSON
    # ---------------------------------------------------------------
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nEvaluation summary saved to {summary_path}")

    # ---------------------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------------------
    if summary_rows:
        print("\n" + "="*65)
        print(f"{'Model':<12} {'AUC':>8} {'1/εB@50%':>12} "
              f"{'ms/batch':>10} {'Params':>12}")
        print("-"*65)
        for row in summary_rows:
            print(f"{row['model']:<12} {row['auc']:>8.4f} "
                  f"{row['rejection_at_50pct_sig_eff']:>12.1f} "
                  f"{row['ms_per_batch']:>10.2f} "
                  f"{row['n_params']:>12,}")
        print("="*65)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate()
