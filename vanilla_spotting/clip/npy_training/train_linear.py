#!/usr/bin/env python3
"""
Training script for pre-extracted npy features (ETRI dataset).

Uses FeatureSpottingModel (temporal aggregation + classification head only).
No vision encoder needed — features are already extracted per-second.

Key Features:
- Lightweight model (only head parameters)
- Class weight balancing
- AUC-ROC as primary metric
- Cosine annealing scheduler with warmup
- Early stopping based on validation AUC
- TensorBoard logging
"""

import argparse
import os
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset import NPYFeatureDataset
from model import FeatureSpottingModel
from utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    get_lr,
    AverageMeter,
    EarlyStopping,
)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler=None,
    desc: str = "Training",
) -> Tuple[float, float]:
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc=desc, leave=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{100*correct/total:.1f}%",
            "lr": f"{get_lr(optimizer):.2e}",
        })

    accuracy = correct / total
    return loss_meter.avg, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validation",
) -> Tuple[float, float, float]:
    """Validate model. Returns (avg_loss, accuracy, AUC-ROC)."""
    model.eval()
    loss_meter = AverageMeter()

    all_labels = []
    all_probs = []
    all_preds = []

    pbar = tqdm(data_loader, desc=desc, leave=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        loss_meter.update(loss.item(), inputs.size(0))
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5
        print("Warning: Only one class in validation set, AUC set to 0.5")

    return loss_meter.avg, accuracy, auc


def train(
    feature_dir: str,
    annotation_dir: str,
    dataset_name: str,
    embed_dim: int = 512,
    unit_duration: int = 2,
    overlap_ratio: float = 0.5,
    temporal_agg: str = "mean",
    dropout_rate: float = 0.5,
    batch_size: int = 64,
    num_workers: int = 4,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    patience: int = 5,
    checkpoint_dir: str = "./checkpoints",
    save_name: str = "linear_spotting",
    save_interval: int = 5,
    log_dir: str = "./runs",
    val_split: float = 0.1,
    seed: int = 42,
    resume_ckpt: str = None,
):
    """Main training function."""
    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%y%m%d%H%M")

    # TensorBoard
    run_name = f"{save_name}_{timestamp}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    print(f"TensorBoard log dir: {writer.log_dir}")

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = NPYFeatureDataset(
        feature_dir=feature_dir,
        annotation_dir=annotation_dir,
        unit_duration=unit_duration,
        overlap_ratio=overlap_ratio,
        strict_normal_sampling=True,
        verbose=True,
        seed=seed,
    )

    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    model = FeatureSpottingModel(
        embed_dim=embed_dim,
        num_classes=2,
        dropout_rate=dropout_rate,
        temporal_agg=temporal_agg,
    )
    model = model.to(device)

    # Class weights
    class_weights = full_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_dir = os.path.join(checkpoint_dir, f"{dataset_name}_linear")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    best_auc = 0.0
    early_stopping = EarlyStopping(patience=patience, mode='max')

    # Resume from checkpoint
    if resume_ckpt is not None:
        print(f"\nLoading weights from checkpoint: {resume_ckpt}")
        load_checkpoint(resume_ckpt, model=model, device=device)

    # Training loop
    print("\n" + "=" * 60)
    print("Training FeatureSpottingModel")
    print("=" * 60)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler,
            desc=f"Epoch {epoch+1}/{epochs}",
        )

        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device,
            desc="Validation",
        )

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.3f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.3f}, AUC={val_auc:.4f}")

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("LR", get_lr(optimizer), epoch)


        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(checkpoint_dir, f"{save_name}_best_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_auc, save_path)

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(
                checkpoint_dir, f"{save_name}_ep{epoch+1}.pth",
            )
            save_checkpoint(model, optimizer, epoch, val_loss, val_auc, save_path)

        if early_stopping(val_auc):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final
    final_path = os.path.join(
        checkpoint_dir, f"{dataset_name}_linear/{save_name}_{epochs}ep.pth",
    )
    save_checkpoint(model, optimizer, epochs, val_loss, val_auc, final_path)

    # Log hparams
    writer.add_hparams(
        {
            "embed_dim": embed_dim,
            "temporal_agg": temporal_agg,
            "unit_duration": unit_duration,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout_rate,
            "overlap_ratio": overlap_ratio,
        },
        {
            "hparam/best_auc": best_auc,
            "hparam/final_val_loss": val_loss,
            "hparam/final_val_acc": val_acc,
        },
    )
    writer.close()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"TensorBoard logs: {writer.log_dir}")
    print("=" * 60)

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on pre-extracted npy features (ETRI dataset)"
    )

    # Data
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Path to feature directory (e.g., .../MCi2-S0-6/train)")
    parser.add_argument("--annotation-dir", type=str, required=True,
                        help="Path to annotation directory with *_timestamp.csv files")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Name of the dataset")

    # Model
    parser.add_argument("--embed-dim", type=int, default=512,
                        help="Feature embedding dimension")
    parser.add_argument("--temporal-agg", type=str, default="mean",
                        choices=["mean", "max", "attention"],
                        help="Temporal aggregation method")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")

    # Window
    parser.add_argument("--unit-duration", type=int, default=2,
                        help="Window size in seconds")
    parser.add_argument("--overlap-ratio", type=float, default=0.5,
                        help="Sliding window overlap ratio")

    # Training
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--save-name", type=str, default="feature_spotting",
                        help="Base name for saved models")
    parser.add_argument("--save-interval", type=int, default=3,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log-dir", type=str, default="./runs",
                        help="TensorBoard log directory")

    # Misc
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training Start at {datetime.now()}\n")
    print("=" * 60)
    print("Feature-based Anomaly Action Spotting - Training")
    print("=" * 60)
    print(f"Feature dir: {args.feature_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Temporal agg: {args.temporal_agg}")
    print(f"Unit duration: {args.unit_duration}s")
    print(f"Overlap ratio: {args.overlap_ratio}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    train(
        feature_dir=args.feature_dir,
        annotation_dir=args.annotation_dir,
        dataset_name=args.dataset_name,
        embed_dim=args.embed_dim,
        unit_duration=args.unit_duration,
        overlap_ratio=args.overlap_ratio,
        temporal_agg=args.temporal_agg,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        save_name=args.save_name,
        save_interval=args.save_interval,
        log_dir=args.log_dir,
        val_split=args.val_split,
        seed=args.seed,
        resume_ckpt=args.resume_ckpt,
    )


if __name__ == "__main__":
    main()
