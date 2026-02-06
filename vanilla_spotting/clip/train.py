#!/usr/bin/env python3
"""
Training script for CLIP-based Anomaly Action Spotting Model.

Two-phase training strategy:
- Phase 1: Freeze CLIP backbone, train only classification head
- Phase 2: Unfreeze all, fine-tune with differential learning rates

Key Features:
- Class weight balancing for imbalanced data
- AUC-ROC as primary metric
- Cosine annealing scheduler with warmup
- Early stopping based on validation AUC
- TensorBoard logging (loss, accuracy, AUC, learning rate)
"""

import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np

from config import (
    SpottingModelConfig,
    SpottingDataConfig,
    SpottingTrainConfig,
    CLIP_MODEL_REGISTRY,
    get_spotting_config,
)
from dataset import UCFCrimeSpottingDataset
from model import SpottingModel
from utils import (
    set_seed,
    get_device,
    create_spotting_transform,
    freeze_backbone,
    unfreeze_all,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    compute_class_weights,
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
    """
    Train model for one epoch.

    Returns:
        Tuple of (average loss, accuracy).
    """
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
    """
    Validate model.

    Returns:
        Tuple of (average loss, accuracy, AUC-ROC).
    """
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


def create_optimizer(
    model: SpottingModel,
    train_cfg: SpottingTrainConfig,
    phase: int = 1,
) -> optim.Optimizer:
    """
    Create optimizer for the specified training phase.

    Phase 1: Only head parameters (+ temporal attention if present).
    Phase 2: Differential LR for backbone vs head.
    """
    if phase == 1:
        params = model.get_head_parameters()
        lr = train_cfg.phase1_lr
    else:
        params = [
            {'params': list(model.get_backbone_parameters()),
             'lr': train_cfg.phase2_backbone_lr},
            {'params': model.get_head_parameters(),
             'lr': train_cfg.phase2_head_lr},
        ]
        lr = train_cfg.phase2_head_lr

    if train_cfg.optimizer.lower() == "adamw":
        return optim.AdamW(
            params,
            lr=lr if phase == 1 else train_cfg.phase2_head_lr,
            weight_decay=train_cfg.weight_decay,
        )
    elif train_cfg.optimizer.lower() == "sgd":
        return optim.SGD(
            params,
            lr=lr if phase == 1 else train_cfg.phase2_head_lr,
            momentum=0.9,
            weight_decay=train_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")


def create_scheduler(
    optimizer: optim.Optimizer,
    train_cfg: SpottingTrainConfig,
    steps_per_epoch: int,
    total_epochs: int,
):
    """Create learning rate scheduler with warmup."""
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * train_cfg.warmup_epochs

    if train_cfg.scheduler.lower() == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = None

    return scheduler


def train(
    model_cfg: SpottingModelConfig,
    data_cfg: SpottingDataConfig,
    train_cfg: SpottingTrainConfig,
    resume_ckpt: str = None,
):
    """
    Main training function with two-phase strategy.
    """
    set_seed(train_cfg.seed)
    device = get_device()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%y%m%d%H%M")

    # TensorBoard
    run_name = f"{train_cfg.save_name}_{model_cfg.name}_{timestamp}"
    writer = SummaryWriter(log_dir=os.path.join(train_cfg.log_dir, run_name))
    print(f"TensorBoard log dir: {writer.log_dir}")

    # Create transforms
    train_transform = create_spotting_transform(model_cfg, data_cfg, is_train=True)
    val_transform = create_spotting_transform(model_cfg, data_cfg, is_train=False)

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = UCFCrimeSpottingDataset(
        video_dir=data_cfg.train_root,
        annotation_csv=data_cfg.train_annotation,
        transform=train_transform,
        overlap_ratio=data_cfg.train_overlap_ratio,
        fallback_fps=data_cfg.fallback_fps,
        unit_duration=data_cfg.unit_duration,
        max_videos_per_class=data_cfg.max_videos_per_class,
        verbose=True,
        seed=train_cfg.seed,
    )

    # Split into train/val
    val_size = int(len(full_dataset) * data_cfg.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )

    # Create model
    model = SpottingModel(
        model_name=model_cfg.name,
        pretrained=model_cfg.pretrained,
        num_classes=model_cfg.num_classes,
        embed_dim=model_cfg.embed_dim,
        dropout_rate=model_cfg.dropout_rate,
        temporal_agg=model_cfg.temporal_agg,
        backend=model_cfg.backend,
    )
    model = model.to(device)

    # Class weights for imbalanced data
    class_weights = full_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    best_auc = 0.0
    global_epoch = 0  # continuous epoch counter across phases

    # Resume from checkpoint
    if resume_ckpt is not None:
        print(f"\nLoading weights from checkpoint: {resume_ckpt}")
        load_checkpoint(resume_ckpt, model=model, device=device)

    # ========================================
    # Phase 1: Train head only
    # ========================================
    print("\n" + "=" * 60)
    print("[Phase 1] Training Head Only (CLIP backbone frozen)")
    print("=" * 60)

    freeze_backbone(model)
    params = count_parameters(model)
    print(f"Trainable parameters: {params['trainable']:,} / {params['total']:,}")

    optimizer_p1 = create_optimizer(model, train_cfg, phase=1)
    scheduler_p1 = create_scheduler(
        optimizer_p1, train_cfg,
        len(train_loader),
        train_cfg.phase1_epochs,
    )

    early_stopping = EarlyStopping(patience=train_cfg.patience, mode='max')

    for epoch in range(train_cfg.phase1_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_p1, device,
            scheduler=scheduler_p1,
            desc=f"P1 Epoch {epoch+1}/{train_cfg.phase1_epochs}",
        )

        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device,
            desc="Validation",
        )

        print(f"Phase 1 - Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.3f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.3f}, AUC={val_auc:.4f}")

        # TensorBoard logging
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, global_epoch)
        writer.add_scalar("AUC/val", val_auc, global_epoch)
        writer.add_scalar("LR/head", get_lr(optimizer_p1), global_epoch)
        global_epoch += 1

        if val_auc > best_auc:
            best_auc = val_auc
            if train_cfg.save_best_only:
                save_path = os.path.join(
                    train_cfg.checkpoint_dir,
                    f"{train_cfg.save_name}_best.pth",
                )
                save_checkpoint(model, optimizer_p1, epoch, val_loss, val_auc, save_path)

        if (epoch + 1) % train_cfg.save_interval == 0:
            save_path = os.path.join(
                train_cfg.checkpoint_dir,
                f"{train_cfg.save_name}_p1_ep{epoch + 1}.pth",
            )
            save_checkpoint(model, optimizer_p1, epoch, val_loss, val_auc, save_path)

        if early_stopping(val_auc):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    total_epochs = train_cfg.phase1_epochs
    final_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.save_name}_{total_epochs}ep_{timestamp}.pth",
    )
    save_checkpoint(model, optimizer_p1, total_epochs, val_loss, val_auc, final_path)

    # Log hyperparameters and final metrics
    writer.add_hparams(
        {
            "model": model_cfg.name,
            "temporal_agg": model_cfg.temporal_agg,
            "embed_dim": model_cfg.embed_dim,
            "batch_size": train_cfg.batch_size,
            "optimizer": train_cfg.optimizer,
            "p1_lr": train_cfg.phase1_lr,
            # "p2_backbone_lr": train_cfg.phase2_backbone_lr,
            # "p2_head_lr": train_cfg.phase2_head_lr,
            "dropout": model_cfg.dropout_rate,
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CLIP-based Anomaly Action Spotting Model"
    )

    # Model selection
    parser.add_argument("--model-name", type=str, default="MobileCLIP-S0",
                        help=f"CLIP model name. Registry: {list(CLIP_MODEL_REGISTRY.keys())}")
    parser.add_argument("--mobileclip-weights", type=str, default=None,
                        help="Path to MobileCLIP v1 pretrained .pt file (required for v1 models)")
    parser.add_argument("--temporal-agg", type=str, default="mean",
                        choices=["mean", "max", "attention"],
                        help="Temporal aggregation method")

    # Data paths
    parser.add_argument("--train-root", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/train",
                        help="Path to training videos")
    parser.add_argument("--train-annotation", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp",
                        help="Path to training annotations")

    # Training
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loading workers")
    parser.add_argument("--phase1-epochs", type=int, default=5,
                        help="Epochs for phase 1 (head only)")
    parser.add_argument("--phase2-epochs", type=int, default=10,
                        help="Epochs for phase 2 (full fine-tuning)")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"],
                        help="Optimizer type")
    parser.add_argument("--phase1-lr", type=float, default=1e-3,
                        help="Learning rate for phase 1")
    parser.add_argument("--phase2-backbone-lr", type=float, default=1e-6,
                        help="Backbone LR for phase 2")
    parser.add_argument("--phase2-head-lr", type=float, default=1e-4,
                        help="Head LR for phase 2")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-name", type=str, default="spotting_clip",
                        help="Base name for saved models")
    parser.add_argument("--log-dir", type=str, default="./runs",
                        help="TensorBoard log directory")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--max-videos-per-class", type=int, default=None,
                        help="Max videos per class for balanced training")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    model_cfg = get_spotting_config(args.model_name)
    model_cfg.temporal_agg = args.temporal_agg

    # Override pretrained path for MobileCLIP v1
    if args.mobileclip_weights and model_cfg.backend == "mobileclip_v1":
        model_cfg.pretrained = args.mobileclip_weights
    elif model_cfg.backend == "mobileclip_v1" and not model_cfg.pretrained:
        print("Error: MobileCLIP v1 requires --mobileclip-weights /path/to/weights.pt")
        return

    data_cfg = SpottingDataConfig(
        train_root=args.train_root,
        train_annotation=args.train_annotation,
        max_videos_per_class=args.max_videos_per_class,
    )

    train_cfg = SpottingTrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_backbone_lr=args.phase2_backbone_lr,
        phase2_head_lr=args.phase2_head_lr,
        optimizer=args.optimizer,
        checkpoint_dir=args.checkpoint_dir,
        save_name=args.save_name,
        log_dir=args.log_dir,
        seed=args.seed,
        patience=args.patience,
    )

    print(f"Training Start at {datetime.now()}\n")
    print("=" * 60)
    print("CLIP Anomaly Action Spotting - Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model_name} (temporal_agg={model_cfg.temporal_agg})")
    print(f"Embed dim: {model_cfg.embed_dim}")
    print(f"Num classes: {model_cfg.num_classes}")
    print(f"Input: {model_cfg.num_frames} frames @ {model_cfg.image_size}x{model_cfg.image_size}")
    print(f"Batch size: {train_cfg.batch_size}")
    print(f"Optimizer: {train_cfg.optimizer}")
    print(f"Phase 1: {train_cfg.phase1_epochs} epochs @ lr={train_cfg.phase1_lr}")
    print(f"Phase 2: {train_cfg.phase2_epochs} epochs @ "
          f"backbone_lr={train_cfg.phase2_backbone_lr}, head_lr={train_cfg.phase2_head_lr}")
    if data_cfg.max_videos_per_class is not None:
        print(f"Video balancing: max {data_cfg.max_videos_per_class} videos per class")
    if args.resume_ckpt:
        print(f"Resume checkpoint: {args.resume_ckpt}")
    print("=" * 60)

    train(model_cfg, data_cfg, train_cfg, resume_ckpt=args.resume_ckpt)


if __name__ == "__main__":
    main()
