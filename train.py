"""
train.py
========
Training loop for Quark/Gluon jet GNN classifiers.

Features:
  - AdamW optimizer with OneCycleLR scheduler
  - Model checkpointing: saves best weights based on validation AUC
  - Early stopping to prevent overfitting
  - Per-epoch logging of loss, accuracy, and AUC

Usage:
    python train.py --model edgeconv --epochs 30 --batch_size 128
    python train.py --model gatv2    --epochs 30 --batch_size 64
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import JetGraphDataset, split_dataset
from models import get_model, count_parameters


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EdgeConv or GATv2 on Quark/Gluon jet dataset"
    )
    parser.add_argument("--model",      type=str, default="edgeconv",
                        choices=["edgeconv", "gatv2"],
                        help="GNN architecture to train")
    parser.add_argument("--root",       type=str, default=".",
                        help="Project root directory (contains data/ folder)")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr",         type=float, default=3e-3)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--k",          type=int, default=8,
                        help="Number of kNN neighbours for graph construction")
    parser.add_argument("--max_jets",   type=int, default=None,
                        help="Cap total jets (useful for quick debugging)")
    parser.add_argument("--patience",   type=int, default=7,
                        help="Early stopping patience (epochs without improvement)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.nll_loss(out, batch.y.squeeze())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y.squeeze()).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Validation / test epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)

        labels = batch.y.squeeze()
        loss = F.nll_loss(out, labels)

        probs = torch.exp(out)[:, 1]   # gluon probability (positive class)

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += batch.num_graphs

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)

    return total_loss / total, correct / total, auc


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nLoading dataset...")
    dataset = JetGraphDataset(
        root=args.root,
        k=args.k,
        augment=False,    # augmentation handled per-split below
        max_jets=args.max_jets
    )

    train_data, val_data, test_data = split_dataset(dataset)

    # Enable augmentation only for training subset
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = get_model(args.model, in_channels=3, dropout=args.dropout)
    model = model.to(device)

    n_params = count_parameters(model)
    print(f"\nModel: {args.model.upper()}")
    print(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    # OneCycleLR: starts low, peaks at lr, then anneals → faster convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3
    )

    # ------------------------------------------------------------------
    # Training loop with early stopping and checkpointing
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(args.root, "checkpoints",
                                   f"best_{args.model}.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_auc = 0.0
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(patience={args.patience})...\n")
    print(f"{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} "
          f"{'ValLoss':>9} {'ValAcc':>8} {'ValAUC':>8}  {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_auc = eval_epoch(
            model, val_loader, device)

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>5} {train_loss:>10.4f} {train_acc:>9.4f} "
              f"{val_loss:>9.4f} {val_acc:>8.4f} {val_auc:>8.4f}  "
              f"{current_lr:>10.2e}  [{elapsed:.1f}s]")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
            "val_auc": val_auc
        })

        # Checkpointing — save only if validation AUC improved
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "model_name": args.model,
                "n_params": n_params
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------
    results_dir = os.path.join(args.root, "results")
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, f"history_{args.model}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History:    {history_path}")

    return best_val_auc


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
