#!/usr/bin/env python3
"""
Evaluation script for Anomaly Action Spotting Model.

Metrics:
- Per-segment accuracy
- AUC-ROC (primary metric for anomaly detection)
- Precision, Recall, F1 at optimal threshold
- Per-video aggregated anomaly score

Output:
- Segment-level anomaly probabilities
- Visualization of anomaly scores over time (optional)
- Detailed metrics report
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    SpottingModelConfig,
    SpottingDataConfig,
    get_spotting_config,
    SPOTTING_CLASSES,
)
from dataset import UCFCrimeSpottingDataset
from model import create_spotting_model
from utils import get_device, create_spotting_transform


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate.
        data_loader: Data loader.
        device: Device to run on.

    Returns:
        Dictionary with all predictions and labels.
    """
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    pbar = tqdm(data_loader, desc="Evaluating")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Get predictions
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Anomaly probability
        all_preds.extend(predicted.cpu().numpy())

    return {
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'preds': np.array(all_preds),
    }


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.5,
    avg_loss: float = None,
) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        labels: Ground truth labels.
        probs: Predicted anomaly probabilities.
        preds: Predicted class labels.
        threshold: Threshold for binary classification.
        avg_loss: Average loss value (optional).

    Returns:
        Dictionary with all metrics.
    """
    # Basic metrics
    accuracy = accuracy_score(labels, preds)

    # AUC-ROC
    if len(np.unique(labels)) > 1:
        auc_roc = roc_auc_score(labels, probs)
    else:
        auc_roc = float('nan')

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Per-class metrics (precision, recall, f1, support)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    # Macro average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # Weighted average
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    # Total support
    total_support = len(labels)

    # Find optimal threshold using Youden's J statistic
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Metrics at optimal threshold
        optimal_preds = (probs >= optimal_threshold).astype(int)
        opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
            labels, optimal_preds, average='binary', zero_division=0
        )
    else:
        optimal_threshold = threshold
        opt_precision = opt_recall = opt_f1 = float('nan')

    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'avg_loss': avg_loss,
        'confusion_matrix': cm,
        # Per-class metrics
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        # Macro average
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        # Weighted average
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        # Total support
        'total_support': total_support,
        # Optimal threshold metrics
        'optimal_threshold': optimal_threshold,
        'opt_precision': opt_precision,
        'opt_recall': opt_recall,
        'opt_f1': opt_f1,
    }


def aggregate_by_video(
    dataset: UCFCrimeSpottingDataset,
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Dict]:
    """
    Aggregate segment-level predictions by video.

    Args:
        dataset: Dataset with sample information.
        probs: Segment-level anomaly probabilities.
        labels: Segment-level labels.

    Returns:
        Dictionary mapping video paths to aggregated scores.
    """
    video_scores = {}

    for idx, (prob, label) in enumerate(zip(probs, labels)):
        sample = dataset.get_sample_info(idx)
        video_path = sample['video_path']

        if video_path not in video_scores:
            video_scores[video_path] = {
                'probs': [],
                'labels': [],
                'timestamps': [],
            }

        video_scores[video_path]['probs'].append(prob)
        video_scores[video_path]['labels'].append(label)
        video_scores[video_path]['timestamps'].append(sample['start_time'])

    # Compute aggregated metrics per video
    for video_path in video_scores:
        probs_arr = np.array(video_scores[video_path]['probs'])
        labels_arr = np.array(video_scores[video_path]['labels'])

        video_scores[video_path].update({
            'max_prob': np.max(probs_arr),
            'mean_prob': np.mean(probs_arr),
            'num_segments': len(probs_arr),
            'num_anomaly_segments': int(np.sum(labels_arr)),
            'video_label': 1 if np.any(labels_arr == 1) else 0,
        })

    return video_scores


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str = "roc_curve.png",
):
    """Plot and save ROC curve."""
    if len(np.unique(labels)) <= 1:
        print("Cannot plot ROC curve: only one class present")
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Anomaly Action Spotting', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str = "pr_curve.png",
):
    """Plot and save Precision-Recall curve."""
    if len(np.unique(labels)) <= 1:
        print("Cannot plot PR curve: only one class present")
        return

    precision, recall, _ = precision_recall_curve(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Anomaly Action Spotting', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PR curve saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str = "confusion_matrix.png",
):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    classes = SPOTTING_CLASSES
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    # Emphasize diagonal cells with border lines
    from matplotlib.patches import Rectangle
    for i in range(min(cm.shape[0], cm.shape[1])):
        rect = Rectangle((i - 0.5, i - 0.5), 1, 1,
                          fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix - Anomaly Action Spotting', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_video_timeline(
    video_scores: Dict,
    video_path: str,
    save_dir: str = "./plots",
):
    """Plot anomaly scores over time for a video."""
    if video_path not in video_scores:
        print(f"Video not found: {video_path}")
        return

    data = video_scores[video_path]
    timestamps = np.array(data['timestamps'])
    probs = np.array(data['probs'])
    labels = np.array(data['labels'])

    plt.figure(figsize=(12, 4))

    # Plot anomaly scores
    plt.plot(timestamps, probs, 'b-', linewidth=1, alpha=0.7, label='Anomaly Score')

    # Highlight ground truth anomaly regions
    anomaly_mask = labels == 1
    if np.any(anomaly_mask):
        plt.fill_between(
            timestamps, 0, 1,
            where=anomaly_mask,
            alpha=0.2, color='red',
            label='Ground Truth Anomaly'
        )

    plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title(f'Anomaly Detection Timeline: {os.path.basename(video_path)}', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"timeline_{os.path.basename(video_path)}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Timeline plot saved to {save_path}")


def print_metrics_report(metrics: Dict):
    """Print formatted metrics report."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n[Segment-Level Metrics]")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  Precision:   {metrics['precision_weighted']:.4f}")
    print(f"  Recall:      {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score:    {metrics['f1_weighted']:.4f}")

    print(f"\n[Optimal Threshold: {metrics['optimal_threshold']:.4f}]")
    print(f"  Precision:   {metrics['opt_precision']:.4f}")
    print(f"  Recall:      {metrics['opt_recall']:.4f}")
    print(f"  F1 Score:    {metrics['opt_f1']:.4f}")

    print(f"\n[Confusion Matrix]")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"              Normal  Abnormal")
    print(f"  Actual Normal   {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"       Abnormal   {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    print("\n" + "=" * 60)


def evaluate(
    checkpoint_path: str,
    data_cfg: SpottingDataConfig,
    model_cfg: SpottingModelConfig = None,
    output_dir: str = "./evaluation",
    plot_videos: int = 5,
):
    """
    Main evaluation function.

    Args:
        checkpoint_path: Path to model checkpoint.
        data_cfg: Data configuration.
        model_cfg: Model configuration.
        output_dir: Directory to save outputs.
        plot_videos: Number of videos to plot timelines for.
    """
    device = get_device()

    if model_cfg is None:
        model_cfg = get_spotting_config()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create transform and dataset
    transform = create_spotting_transform(model_cfg, data_cfg, is_train=False)

    dataset = UCFCrimeSpottingDataset(
        video_dir=data_cfg.test_root,
        annotation_csv=data_cfg.test_annotation,
        transform=transform,
        overlap_ratio=data_cfg.infer_overlap_ratio,  # No overlap for inference
        fallback_fps=data_cfg.fallback_fps,
        unit_duration=data_cfg.unit_duration,
        max_videos_per_class=data_cfg.max_videos_per_class,
        verbose=True,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    model = create_spotting_model(
        num_classes=model_cfg.num_classes,
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(model, data_loader, device)

    # Compute metrics
    metrics = compute_metrics(results['labels'], results['probs'], results['preds'])

    # Print report
    print_metrics_report(metrics)

    # Save plots
    plot_roc_curve(results['labels'], results['probs'],
                   os.path.join(output_dir, "roc_curve.png"))
    plot_precision_recall_curve(results['labels'], results['probs'],
                                os.path.join(output_dir, "pr_curve.png"))
    plot_confusion_matrix(metrics['confusion_matrix'],
                          os.path.join(output_dir, "confusion_matrix.png"))

    # Aggregate by video
    video_scores = aggregate_by_video(dataset, results['probs'], results['labels'])

    # Plot video timelines
    if plot_videos > 0:
        # Select videos with most anomaly segments
        sorted_videos = sorted(
            video_scores.keys(),
            key=lambda x: video_scores[x]['num_anomaly_segments'],
            reverse=True
        )
        for video_path in sorted_videos[:plot_videos]:
            plot_video_timeline(video_scores, video_path,
                                os.path.join(output_dir, "timelines"))

    # Save results
    np.savez(
        os.path.join(output_dir, "results.npz"),
        labels=results['labels'],
        probs=results['probs'],
        preds=results['preds'],
    )
    print(f"\nResults saved to {output_dir}")

    return metrics, video_scores


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Anomaly Action Spotting Model"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test-root", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/test",
                        help="Path to test videos")
    parser.add_argument("--test-annotation", type=str,
                        default="/mnt/c/JJS/UCF_Crimes/Videos/test/00_timestamp",
                        help="Path to test annotations")
    parser.add_argument("--output-dir", type=str, default="./evaluation",
                        help="Directory to save outputs")
    parser.add_argument("--plot-videos", type=int, default=5,
                        help="Number of videos to plot timelines for")
    parser.add_argument("--max-videos-per-class", type=int, default=None,
                        help="Max videos per class for balanced evaluation (default: None = no limit)")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    data_cfg = SpottingDataConfig(
        test_root=args.test_root,
        test_annotation=args.test_annotation,
        max_videos_per_class=args.max_videos_per_class,
    )

    evaluate(
        checkpoint_path=args.checkpoint,
        data_cfg=data_cfg,
        output_dir=args.output_dir,
        plot_videos=args.plot_videos,
    )


if __name__ == "__main__":
    main()
