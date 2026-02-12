#!/usr/bin/env python3
"""
Video-level binary classification evaluation script.

Evaluates video-level (file-level) anomaly detection performance
using pre-extracted npy features.

Metrics:
- Video-level Accuracy
- Video-level AUC-ROC
- Precision, Recall, F1 at various thresholds
- Confusion Matrix

Usage:
    python evaluate_video.py \
        --checkpoint checkpoints/model.pth \
        --feature-dir /path/to/features \
        --annotation-dir /path/to/annotations \
        --model-type resmlp
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset import NPYFeatureDataset
from model import FeatureSpottingModel, ResidualMLPSpottingModel
from utils import set_seed, get_device, load_checkpoint


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    model_type: str = "resmlp",
) -> Dict:
    """
    Run inference and return segment-level predictions.

    Returns:
        Dictionary with labels, probs, preds arrays.
    """
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    pbar = tqdm(data_loader, desc="Running inference")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        outputs = model(inputs)

        if model_type == "resmlp":
            logits = outputs.squeeze(1)
            probs = torch.sigmoid(logits)
            predicted = (logits > 0).long()
        else:
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    return {
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'preds': np.array(all_preds),
    }


def aggregate_to_video_level(
    dataset: NPYFeatureDataset,
    probs: np.ndarray,
    labels: np.ndarray,
    agg_method: str = "max",
) -> Dict[str, Dict]:
    """
    Aggregate segment-level predictions to video (file) level.

    Args:
        dataset: NPYFeatureDataset instance.
        probs: Segment-level anomaly probabilities.
        labels: Segment-level ground truth labels.
        agg_method: Aggregation method ("max", "mean").

    Returns:
        Dictionary mapping npy_path to aggregated scores.
    """
    file_scores = {}

    for idx, (prob, label) in enumerate(zip(probs, labels)):
        sample = dataset.get_sample_info(idx)
        npy_path = sample['npy_path']

        if npy_path not in file_scores:
            file_scores[npy_path] = {
                'probs': [],
                'labels': [],
            }

        file_scores[npy_path]['probs'].append(prob)
        file_scores[npy_path]['labels'].append(label)

    # Aggregate
    for npy_path in file_scores:
        probs_arr = np.array(file_scores[npy_path]['probs'])
        labels_arr = np.array(file_scores[npy_path]['labels'])

        if agg_method == "max":
            agg_prob = np.max(probs_arr)
        elif agg_method == "mean":
            agg_prob = np.mean(probs_arr)
        else:
            agg_prob = np.max(probs_arr)

        file_scores[npy_path].update({
            'video_prob': agg_prob,
            'video_label': 1 if np.any(labels_arr == 1) else 0,
            'num_segments': len(probs_arr),
            'num_anomaly_segments': int(np.sum(labels_arr)),
            'class_name': os.path.basename(os.path.dirname(npy_path)),
        })

    return file_scores


def compute_video_metrics(
    file_scores: Dict[str, Dict],
    threshold: float = 0.5,
) -> Dict:
    """
    Compute video-level evaluation metrics.

    Args:
        file_scores: Dictionary from aggregate_to_video_level().
        threshold: Threshold for binary prediction.

    Returns:
        Dictionary with video-level metrics.
    """
    video_labels = []
    video_probs = []
    video_preds = []

    for npy_path, data in file_scores.items():
        video_labels.append(data['video_label'])
        video_probs.append(data['video_prob'])
        video_preds.append(1 if data['video_prob'] >= threshold else 0)

    video_labels = np.array(video_labels)
    video_probs = np.array(video_probs)
    video_preds = np.array(video_preds)

    # Basic metrics
    accuracy = accuracy_score(video_labels, video_preds)

    # AUC-ROC
    if len(np.unique(video_labels)) > 1:
        auc_roc = roc_auc_score(video_labels, video_probs)
    else:
        auc_roc = float('nan')

    # Confusion matrix
    cm = confusion_matrix(video_labels, video_preds)

    # Precision, Recall, F1
    if len(np.unique(video_labels)) > 1 and len(np.unique(video_preds)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            video_labels, video_preds, average='binary', zero_division=0
        )
    else:
        precision = recall = f1 = float('nan')

    # Optimal threshold via Youden's J
    if len(np.unique(video_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(video_labels, video_probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        optimal_preds = (video_probs >= optimal_threshold).astype(int)
        opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
            video_labels, optimal_preds, average='binary', zero_division=0
        )
        opt_accuracy = accuracy_score(video_labels, optimal_preds)
    else:
        optimal_threshold = threshold
        opt_precision = opt_recall = opt_f1 = opt_accuracy = float('nan')

    # Counts
    n_total = len(video_labels)
    n_anomaly = int(np.sum(video_labels))
    n_normal = n_total - n_anomaly

    tp = int(np.sum((video_preds == 1) & (video_labels == 1)))
    fp = int(np.sum((video_preds == 1) & (video_labels == 0)))
    tn = int(np.sum((video_preds == 0) & (video_labels == 0)))
    fn = int(np.sum((video_preds == 0) & (video_labels == 1)))

    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'threshold': threshold,
        'optimal_threshold': optimal_threshold,
        'opt_accuracy': opt_accuracy,
        'opt_precision': opt_precision,
        'opt_recall': opt_recall,
        'opt_f1': opt_f1,
        'n_total': n_total,
        'n_anomaly': n_anomaly,
        'n_normal': n_normal,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'video_labels': video_labels,
        'video_probs': video_probs,
        'video_preds': video_preds,
    }


def print_video_metrics(metrics: Dict, agg_method: str):
    """Print video-level evaluation report."""
    print("\n" + "=" * 70)
    print("VIDEO-LEVEL BINARY CLASSIFICATION EVALUATION")
    print("=" * 70)

    print(f"\nAggregation Method: {agg_method.upper()}")
    print(f"Total Videos: {metrics['n_total']} "
          f"(Anomaly: {metrics['n_anomaly']}, Normal: {metrics['n_normal']})")

    print(f"\n[Metrics @ threshold={metrics['threshold']:.2f}]")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics['auc_roc']) else "N/A"
    print(f"  AUC-ROC:     {auc_str}")
    prec_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "N/A"
    rec_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "N/A"
    f1_str = f"{metrics['f1']:.4f}" if not np.isnan(metrics['f1']) else "N/A"
    print(f"  Precision:   {prec_str}")
    print(f"  Recall:      {rec_str}")
    print(f"  F1 Score:    {f1_str}")

    print(f"\n[Optimal Threshold: {metrics['optimal_threshold']:.4f}]")
    opt_acc_str = f"{metrics['opt_accuracy']:.4f}" if not np.isnan(metrics['opt_accuracy']) else "N/A"
    opt_prec_str = f"{metrics['opt_precision']:.4f}" if not np.isnan(metrics['opt_precision']) else "N/A"
    opt_rec_str = f"{metrics['opt_recall']:.4f}" if not np.isnan(metrics['opt_recall']) else "N/A"
    opt_f1_str = f"{metrics['opt_f1']:.4f}" if not np.isnan(metrics['opt_f1']) else "N/A"
    print(f"  Accuracy:    {opt_acc_str}")
    print(f"  Precision:   {opt_prec_str}")
    print(f"  Recall:      {opt_rec_str}")
    print(f"  F1 Score:    {opt_f1_str}")

    print(f"\n[Confusion Matrix]")
    print(f"                Predicted")
    print(f"              Normal  Anomaly")
    print(f"  Actual Normal   {metrics['tn']:5d}   {metrics['fp']:5d}")
    print(f"       Anomaly    {metrics['fn']:5d}   {metrics['tp']:5d}")

    print("\n" + "=" * 70)


def compute_per_class_video_metrics(file_scores: Dict[str, Dict], threshold: float = 0.5) -> Dict[str, Dict]:
    """
    Compute video-level metrics per class (parent directory).

    Returns:
        Dictionary mapping class_name to metrics.
    """
    from collections import defaultdict

    class_data = defaultdict(lambda: {'labels': [], 'probs': [], 'preds': []})

    for npy_path, data in file_scores.items():
        class_name = data['class_name']
        class_data[class_name]['labels'].append(data['video_label'])
        class_data[class_name]['probs'].append(data['video_prob'])
        class_data[class_name]['preds'].append(1 if data['video_prob'] >= threshold else 0)

    class_metrics = {}
    for class_name, data in sorted(class_data.items()):
        labels = np.array(data['labels'])
        probs = np.array(data['probs'])
        preds = np.array(data['preds'])

        n_videos = len(labels)
        n_anomaly = int(np.sum(labels))
        n_normal = n_videos - n_anomaly

        accuracy = np.mean(preds == labels)

        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
        else:
            auc = float('nan')

        class_metrics[class_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'n_videos': n_videos,
            'n_anomaly': n_anomaly,
            'n_normal': n_normal,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }

    return class_metrics


def print_per_class_video_report(class_metrics: Dict[str, Dict]):
    """Print per-class video-level metrics report."""
    print("\n" + "=" * 85)
    print("PER-CLASS VIDEO-LEVEL METRICS")
    print("=" * 85)

    print(f"{'Class':<15} {'Videos':>7} {'Anom':>5} {'Norm':>5} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 85)

    # Sort: anomaly classes first, then Normal
    anomaly_classes = [(k, v) for k, v in class_metrics.items() if k.lower() != 'normal']
    normal_classes = [(k, v) for k, v in class_metrics.items() if k.lower() == 'normal']

    anomaly_classes.sort(key=lambda x: x[1]['accuracy'])
    all_classes = anomaly_classes + normal_classes

    for class_name, m in all_classes:
        auc_str = f"{m['auc']:.4f}" if not np.isnan(m['auc']) else "N/A"
        print(f"{class_name:<15} {m['n_videos']:>7} {m['n_anomaly']:>5} {m['n_normal']:>5} "
              f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {auc_str:>7}")

    print("=" * 85)


def plot_video_roc(metrics: Dict, save_path: str):
    """Plot and save video-level ROC curve."""
    labels = metrics['video_labels']
    probs = metrics['video_probs']

    if len(np.unique(labels)) <= 1:
        print("Cannot plot ROC curve: only one class present")
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = metrics['auc_roc']

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Video-Level AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

    # Mark optimal threshold point
    opt_idx = np.argmax(tpr - fpr)
    plt.scatter([fpr[opt_idx]], [tpr[opt_idx]], c='red', s=100, zorder=5,
                label=f'Optimal (thresh={metrics["optimal_threshold"]:.3f})')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Video-Level ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_path}")


def evaluate_video(
    checkpoint_path: str,
    feature_dir: str,
    annotation_dir: str,
    model_type: str = "resmlp",
    embed_dim: int = 512,
    temporal_agg: str = "mean",
    dropout_rate: float = 0.5,
    unit_duration: int = 2,
    overlap_ratio: float = 0.5,
    agg_method: str = "max",
    threshold: float = 0.5,
    batch_size: int = 64,
    num_workers: int = 4,
    output_dir: str = "./evaluation_video",
    seed: int = 42,
):
    """Main video-level evaluation function."""
    set_seed(seed)
    device = get_device()

    # Load dataset
    print("\nLoading dataset...")
    dataset = NPYFeatureDataset(
        feature_dir=feature_dir,
        annotation_dir=annotation_dir,
        unit_duration=unit_duration,
        overlap_ratio=overlap_ratio,
        strict_normal_sampling=False,
        verbose=True,
        seed=seed,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load model
    if model_type == "resmlp":
        model = ResidualMLPSpottingModel(
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
            temporal_agg=temporal_agg,
        )
    else:
        model = FeatureSpottingModel(
            embed_dim=embed_dim,
            num_classes=2,
            dropout_rate=dropout_rate,
            temporal_agg=temporal_agg,
        )

    print(f"\nLoading checkpoint: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model=model, device=device)
    model = model.to(device)

    # Run inference
    print("\nRunning inference...")
    results = run_inference(model, data_loader, device, model_type=model_type)

    # Aggregate to video level
    print("\nAggregating to video level...")
    file_scores = aggregate_to_video_level(
        dataset, results['probs'], results['labels'], agg_method=agg_method
    )

    # Compute metrics
    metrics = compute_video_metrics(file_scores, threshold=threshold)
    print_video_metrics(metrics, agg_method)

    # Per-class metrics
    class_metrics = compute_per_class_video_metrics(file_scores, threshold=threshold)
    print_per_class_video_report(class_metrics)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    plot_video_roc(metrics, os.path.join(output_dir, "video_roc_curve.png"))

    np.savez(
        os.path.join(output_dir, "video_results.npz"),
        video_labels=metrics['video_labels'],
        video_probs=metrics['video_probs'],
        video_preds=metrics['video_preds'],
    )
    print(f"\nResults saved to {output_dir}")

    return metrics, file_scores, class_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video-level binary classification evaluation"
    )

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Path to feature directory")
    parser.add_argument("--annotation-dir", type=str, required=True,
                        help="Path to annotation directory")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["linear", "resmlp"],
                        help="Model type")

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

    # Aggregation
    parser.add_argument("--agg-method", type=str, default="max",
                        choices=["max", "mean"],
                        help="Video-level aggregation method")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary classification threshold")

    # Evaluation
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--output-dir", type=str, default="./evaluation_video",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Video-Level Binary Classification Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Feature dir: {args.feature_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Aggregation method: {args.agg_method}")
    print(f"Threshold: {args.threshold}")
    print("=" * 70)

    evaluate_video(
        checkpoint_path=args.checkpoint,
        feature_dir=args.feature_dir,
        annotation_dir=args.annotation_dir,
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        temporal_agg=args.temporal_agg,
        dropout_rate=args.dropout,
        unit_duration=args.unit_duration,
        overlap_ratio=args.overlap_ratio,
        agg_method=args.agg_method,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
