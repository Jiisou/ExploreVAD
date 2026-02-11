#!/usr/bin/env python3
"""
Training script for TemporalSelfAttentionModel on pre-extracted frame-level features.

Architecture:
    (B, T, D) → Frame Skip → Sliding Window Self-Attention → (B, D)
    → x + MLP(x) [residual, QuickGELU]
    → Linear(D, 1) → BCEWithLogitsLoss

Key features:
- Frame-level features with duplication handling (frame_skip)
- Self-attention within sliding windows
- 50% overlapping windows (1 second stride for 2 second window)
- ResidualMLP classification head (CLIP-style)
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

from dataset import FrameLevelFeatureDataset
from model import TemporalSelfAttentionModel, SimplifiedTemporalAttention


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 5, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_lr(optimizer: optim.Optimizer) -> float:
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    auc: float,
    path: str,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'auc': auc,
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    device: torch.device = None,
):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('auc', 0)


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
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(inputs).squeeze(-1)  # (B,)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), inputs.size(0))
        predicted = (logits > 0).long()
        total += labels.size(0)
        correct += predicted.eq(labels.long()).sum().item()

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
        labels_float = labels.float().to(device)

        logits = model(inputs).squeeze(-1)
        loss = criterion(logits, labels_float)

        probs = torch.sigmoid(logits)
        predicted = (logits > 0).long()

        loss_meter.update(loss.item(), inputs.size(0))
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
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


def compute_pos_weight(dataset: FrameLevelFeatureDataset) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss."""
    labels = [s['label'] for s in dataset.samples]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    if num_pos == 0:
        print("Warning: No positive samples found, pos_weight=1.0")
        return torch.tensor([1.0])

    pw = num_neg / num_pos
    print(f"Class distribution: neg={num_neg}, pos={num_pos}")
    print(f"pos_weight: {pw:.4f}")
    return torch.tensor([pw])


def train(
    feature_dir: str,
    annotation_dir: str,
    dataset_name: str,
    embed_dim: int = 512,
    fps: int = 6,
    frame_skip: int = 6,
    window_seconds: float = 2.0,
    overlap_ratio: float = 0.5,
    num_heads: int = 4,
    num_layers: int = 2,
    window_agg: str = "mean",
    dropout: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 4,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    patience: int = 5,
    checkpoint_dir: str = "./checkpoints",
    save_name: str = "temporal_attn",
    save_interval: int = 5,
    log_dir: str = "./runs",
    val_split: float = 0.1,
    seed: int = 42,
    resume_ckpt: str = None,
    use_simplified: bool = False,
):
    """Main training function."""
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%y%m%d%H%M")

    # TensorBoard
    run_name = f"{save_name}_{timestamp}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    print(f"TensorBoard log dir: {writer.log_dir}")

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = FrameLevelFeatureDataset(
        feature_dir=feature_dir,
        annotation_dir=annotation_dir,
        fps=fps,
        frame_skip=frame_skip,
        window_seconds=window_seconds,
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

    # Calculate window_frames for model (unique frames after frame_skip)
    # e.g., 30fps video, 2s window, 6 frame skip → 60 raw / 6 = 10 unique frames
    window_frames = int(window_seconds * fps / frame_skip)

    # Create model
    if use_simplified:
        model = SimplifiedTemporalAttention(
            embed_dim=embed_dim,
            window_frames=window_frames,
            frame_skip=frame_skip,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            dropout=dropout,
        )
    else:
        model = TemporalSelfAttentionModel(
            embed_dim=embed_dim,
            window_frames=window_frames,
            frame_skip=frame_skip,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            window_agg=window_agg,
        )
    model = model.to(device)

    # Print trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # pos_weight for class imbalance
    pos_weight = compute_pos_weight(full_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    checkpoint_save_dir = os.path.join(checkpoint_dir, f"{dataset_name}_temporal")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

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
    print("Training TemporalSelfAttentionModel")
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
            save_path = os.path.join(checkpoint_save_dir, f"{save_name}_best_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_auc, save_path)

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(checkpoint_save_dir, f"{save_name}_ep{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_auc, save_path)

        if early_stopping(val_auc):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final
    final_path = os.path.join(checkpoint_save_dir, f"{save_name}_final.pth")
    save_checkpoint(model, optimizer, epochs, val_loss, val_auc, final_path)

    # Log hparams
    writer.add_hparams(
        {
            "embed_dim": embed_dim,
            "fps": fps,
            "frame_skip": frame_skip,
            "window_seconds": window_seconds,
            "overlap_ratio": overlap_ratio,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "window_agg": window_agg,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
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
        description="Train TemporalSelfAttentionModel on frame-level features"
    )

    # Data
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Path to feature directory")
    parser.add_argument("--annotation-dir", type=str, required=True,
                        help="Path to annotation directory with *_timestamp.csv files")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Name of the dataset ['UCF' or 'ETRI']")

    # Model
    parser.add_argument("--embed-dim", type=int, default=512,
                        help="Feature embedding dimension")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of transformer layers per window")
    parser.add_argument("--window-agg", type=str, default="attention",
                        choices=["mean", "max", "attention"],
                        help="Window aggregation method")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--use-simplified", action="store_true",
                        help="Use simplified model (whole sequence into [CLS])")

    # Feature parameters
    parser.add_argument("--fps", type=int, default=30,
                        help="Feature extraction FPS (raw features per second in npy)")
    parser.add_argument("--frame-skip", type=int, default=6,
                        help="Number of duplicated frames to skip")
    parser.add_argument("--window-seconds", type=float, default=2.0,
                        help="Window duration in seconds")
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
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="./temporal_modeling/checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--save-name", type=str, default="temporal_attn",
                        help="Base name for saved models")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log-dir", type=str, default="./temporal_modeling/runs",
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
    print("Temporal Self-Attention Model Training")
    print("=" * 60)
    print(f"Feature dir: {args.feature_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"FPS: {args.fps}, Frame skip: {args.frame_skip}")
    print(f"Window: {args.window_seconds}s, Overlap: {args.overlap_ratio}")
    print(f"Attention heads: {args.num_heads}, Layers: {args.num_layers}")
    print(f"Window aggregation: {args.window_agg}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    train(
        feature_dir=args.feature_dir,
        annotation_dir=args.annotation_dir,
        dataset_name=args.dataset_name,
        embed_dim=args.embed_dim,
        fps=args.fps,
        frame_skip=args.frame_skip,
        window_seconds=args.window_seconds,
        overlap_ratio=args.overlap_ratio,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        window_agg=args.window_agg,
        dropout=args.dropout,
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
        use_simplified=args.use_simplified,
    )


if __name__ == "__main__":
    main()
