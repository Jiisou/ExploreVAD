#!/usr/bin/env python3
"""
Evaluation script for pre-extracted npy features.

Uses FeatureSpottingModel (temporal aggregation + classification head only).

Metrics:
- Per-segment accuracy
- AUC-ROC (primary metric)
- Precision, Recall, F1 at optimal threshold
- Per-file aggregated anomaly score
- Timeline plots with normal/anomaly logits
"""

import argparse
import os
from typing import Dict, List, Tuple
from collections import defaultdict

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
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset import NPYFeatureDataset
from model import FeatureSpottingModel, ResidualMLPSpottingModel
from utils import set_seed, get_device, load_checkpoint
from config import SPOTTING_CLASSES

# Frame-level constants
DEFAULT_FPS = 30  # Default frames per second


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    model_type: str = "linear",
) -> Dict:
    """
    Evaluate model on dataset.

    Args:
        model_type: "linear" (FeatureSpottingModel, 2-class softmax) or
                    "resmlp" (ResidualMLPSpottingModel, single logit sigmoid).

    Returns:
        Dictionary with labels, probs, logits, and preds arrays.
    """
    model.eval()

    all_labels = []
    all_probs = []
    all_logits_normal = []
    all_logits_anomaly = []
    all_preds = []

    pbar = tqdm(data_loader, desc="Evaluating")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        outputs = model(inputs)

        if model_type == "resmlp":
            # Single logit output: (B, 1)
            logits = outputs.squeeze(1)  # (B,)
            probs = torch.sigmoid(logits)
            predicted = (logits > 0).long()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits_anomaly.extend(logits.cpu().numpy())
            all_logits_normal.extend((-logits).cpu().numpy())  # implicit normal logit
            all_preds.extend(predicted.cpu().numpy())
        else:
            # 2-class output: (B, 2)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_logits_normal.extend(outputs[:, 0].cpu().numpy())
            all_logits_anomaly.extend(outputs[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return {
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'logits_normal': np.array(all_logits_normal),
        'logits_anomaly': np.array(all_logits_anomaly),
        'preds': np.array(all_preds),
    }


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute evaluation metrics."""
    accuracy = accuracy_score(labels, preds)

    if len(np.unique(labels)) > 1:
        auc_roc = roc_auc_score(labels, probs)
    else:
        auc_roc = float('nan')

    cm = confusion_matrix(labels, preds)

    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    total_support = len(labels)

    # # Optimal threshold via Youden's J statistic
    # if len(np.unique(labels)) > 1:
    #     fpr, tpr, thresholds = roc_curve(labels, probs)
    #     j_scores = tpr - fpr
    #     optimal_idx = np.argmax(j_scores)
    #     optimal_threshold = thresholds[optimal_idx]

    #     optimal_preds = (probs >= optimal_threshold).astype(int)
    #     opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
    #         labels, optimal_preds, average='binary', zero_division=0
    #     )
    # else:
    #     optimal_threshold = threshold
    #     opt_precision = opt_recall = opt_f1 = float('nan')

    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'total_support': total_support,
        # 'optimal_threshold': optimal_threshold,
        # 'opt_precision': opt_precision,
        # 'opt_recall': opt_recall,
        # 'opt_f1': opt_f1,
    }


def compute_per_class_metrics(
    dataset: NPYFeatureDataset,
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Dict]:
    """
    Compute binary prediction metrics per anomaly class.

    Groups samples by parent directory name (class) and computes
    accuracy, precision, recall, F1 for each class.

    Args:
        dataset: NPYFeatureDataset instance.
        preds: Predicted labels (0 or 1).
        labels: Ground truth labels (0 or 1).
        probs: Anomaly probabilities.

    Returns:
        Dictionary mapping class_name to metrics dict.
    """
    from collections import defaultdict

    # Group by class
    class_data: Dict[str, Dict] = defaultdict(lambda: {
        'preds': [], 'labels': [], 'probs': []
    })

    for idx in range(len(preds)):
        sample = dataset.get_sample_info(idx)
        npy_path = sample['npy_path']
        class_name = os.path.basename(os.path.dirname(npy_path))

        class_data[class_name]['preds'].append(preds[idx])
        class_data[class_name]['labels'].append(labels[idx])
        class_data[class_name]['probs'].append(probs[idx])

    # Compute metrics per class
    class_metrics = {}
    for class_name, data in sorted(class_data.items()):
        preds_arr = np.array(data['preds'])
        labels_arr = np.array(data['labels'])
        probs_arr = np.array(data['probs'])

        n_samples = len(preds_arr)
        n_anomaly_gt = int(np.sum(labels_arr))
        n_normal_gt = n_samples - n_anomaly_gt

        # Binary accuracy
        accuracy = np.mean(preds_arr == labels_arr)

        # True Positives, False Positives, etc.
        tp = int(np.sum((preds_arr == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds_arr == 1) & (labels_arr == 0)))
        tn = int(np.sum((preds_arr == 0) & (labels_arr == 0)))
        fn = int(np.sum((preds_arr == 0) & (labels_arr == 1)))

        # Precision, Recall, F1 for anomaly class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AUC if both classes present
        if len(np.unique(labels_arr)) > 1:
            auc = roc_auc_score(labels_arr, probs_arr)
        else:
            auc = float('nan')

        class_metrics[class_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'n_samples': n_samples,
            'n_anomaly_gt': n_anomaly_gt,
            'n_normal_gt': n_normal_gt,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }

    return class_metrics


def print_per_class_report(class_metrics: Dict[str, Dict]):
    """Print per-class binary prediction accuracy report."""
    print("\n" + "=" * 80)
    print("PER-CLASS BINARY PREDICTION METRICS")
    print("=" * 80)

    # Header
    print(f"{'Class':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'AUC':>7} {'Samples':>8} {'GT_Anom':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 80)

    # Sort: anomaly classes first (by accuracy), then Normal
    anomaly_classes = [(k, v) for k, v in class_metrics.items() if k.lower() != 'normal']
    normal_class = [(k, v) for k, v in class_metrics.items() if k.lower() == 'normal']

    # Sort anomaly classes by accuracy (ascending to see worst first)
    anomaly_classes.sort(key=lambda x: x[1]['accuracy'])

    all_classes = anomaly_classes + normal_class

    for class_name, m in all_classes:
        auc_str = f"{m['auc']:.4f}" if not np.isnan(m['auc']) else "N/A"
        print(f"{class_name:<15} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {auc_str:>7} "
              f"{m['n_samples']:>8} {m['n_anomaly_gt']:>8} "
              f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    # Summary statistics
    print("-" * 80)

    # Macro average (excluding Normal for anomaly detection focus)
    anomaly_only = [m for k, m in class_metrics.items() if k.lower() != 'normal']
    if anomaly_only:
        macro_acc = np.mean([m['accuracy'] for m in anomaly_only])
        macro_prec = np.mean([m['precision'] for m in anomaly_only])
        macro_rec = np.mean([m['recall'] for m in anomaly_only])
        macro_f1 = np.mean([m['f1'] for m in anomaly_only])
        aucs = [m['auc'] for m in anomaly_only if not np.isnan(m['auc'])]
        macro_auc = np.mean(aucs) if aucs else float('nan')
        total_samples = sum(m['n_samples'] for m in anomaly_only)

        auc_str = f"{macro_auc:.4f}" if not np.isnan(macro_auc) else "N/A"
        print(f"{'[Anomaly Avg]':<15} {macro_acc:>7.4f} {macro_prec:>7.4f} "
              f"{macro_rec:>7.4f} {macro_f1:>7.4f} {auc_str:>7} {total_samples:>8}")

    print("=" * 80)


def aggregate_by_file(
    dataset: NPYFeatureDataset,
    probs: np.ndarray,
    labels: np.ndarray,
    logits_normal: np.ndarray = None,
    logits_anomaly: np.ndarray = None,
) -> Dict[str, Dict]:
    """Aggregate segment-level predictions by npy file."""
    file_scores = {}

    for idx, (prob, label) in enumerate(zip(probs, labels)):
        sample = dataset.get_sample_info(idx)
        npy_path = sample['npy_path']

        if npy_path not in file_scores:
            file_scores[npy_path] = {
                'probs': [],
                'logits_normal': [],
                'logits_anomaly': [],
                'labels': [],
                'timestamps': [],
            }

        file_scores[npy_path]['probs'].append(prob)
        if logits_normal is not None:
            file_scores[npy_path]['logits_normal'].append(logits_normal[idx])
        if logits_anomaly is not None:
            file_scores[npy_path]['logits_anomaly'].append(logits_anomaly[idx])
        file_scores[npy_path]['labels'].append(label)
        file_scores[npy_path]['timestamps'].append(sample['start_idx'])

    for npy_path in file_scores:
        probs_arr = np.array(file_scores[npy_path]['probs'])
        labels_arr = np.array(file_scores[npy_path]['labels'])

        # Calculate segment-level accuracy and F1 (threshold at 0.5)
        preds_arr = (probs_arr >= 0.5).astype(int)
        segment_acc = np.mean(preds_arr == labels_arr)

        # Segment-level F1 for anomaly class
        tp = int(np.sum((preds_arr == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds_arr == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds_arr == 0) & (labels_arr == 1)))
        seg_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        seg_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        segment_f1 = (2 * seg_precision * seg_recall / (seg_precision + seg_recall)
                       if (seg_precision + seg_recall) > 0 else 0.0)

        file_scores[npy_path].update({
            'max_prob': np.max(probs_arr),
            'mean_prob': np.mean(probs_arr),
            'num_segments': len(probs_arr),
            'num_anomaly_segments': int(np.sum(labels_arr)),
            'file_label': 1 if np.any(labels_arr == 1) else 0,
            'segment_acc': segment_acc,
            'segment_f1': segment_f1,
            'segment_precision': seg_precision,
            'segment_recall': seg_recall,
        })

    return file_scores


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """Plot and save ROC curve."""
    if len(np.unique(labels)) <= 1:
        print("Cannot plot ROC curve: only one class present")
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=1, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Feature Anomaly Spotting ', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """Plot and save Precision-Recall curve."""
    if len(np.unique(labels)) <= 1:
        print("Cannot plot PR curve: only one class present")
        return

    precision, recall, _ = precision_recall_curve(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Feature Anomaly Spotting ', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PR curve saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
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

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    from matplotlib.patches import Rectangle
    for i in range(min(cm.shape[0], cm.shape[1])):
        rect = Rectangle((i - 0.5, i - 0.5), 1, 1,
                          fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix - Feature Anomaly Spotting', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_file_timeline(file_scores: Dict, npy_path: str, save_dir: str = "./timeline_plots"):
    """Plot normal and anomaly logits over time for a npy file."""
    if npy_path not in file_scores:
        print(f"File not found: {npy_path}")
        return

    data = file_scores[npy_path]
    timestamps = np.array(data['timestamps'])
    logits_normal = np.array(data['logits_normal'])
    logits_anomaly = np.array(data['logits_anomaly'])
    labels = np.array(data['labels'])

    all_logits = np.concatenate([logits_normal, logits_anomaly])
    y_min, y_max = all_logits.min(), all_logits.max()
    y_margin = max(0.5, (y_max - y_min) * 0.1)

    plt.figure(figsize=(12, 4))
    # plt.plot(timestamps, logits_normal, 'k-', linewidth=1, alpha=0.7, label='Normal Logit')
    plt.plot(timestamps, logits_anomaly, 'b-', linewidth=2, alpha=0.7, label='Anomaly Logit')
    # smoothed line
    logits_anomaly_smooth = gaussian_filter1d(logits_anomaly, sigma=2)
    plt.plot(timestamps, logits_anomaly_smooth, 'g-', linewidth=1, label='Smoothed (Sigma=3)')

    anomaly_mask = labels == 1
    if np.any(anomaly_mask):
        plt.fill_between(
            timestamps, y_min - y_margin, y_max + y_margin,
            where=anomaly_mask,
            alpha=0.2, color='red',
            label='Ground Truth Anomaly'
        )

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Logit', fontsize=12)

    file_stem = os.path.splitext(os.path.basename(npy_path))[0]
    plt.title(f'Anomaly Timeline: {file_stem}', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"timeline_{file_stem}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Timeline plot saved to {save_path}")


def print_metrics_report(metrics: Dict):
    """Print formatted metrics report."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS ")
    print("=" * 60)

    print(f"\n[Segment-Level Metrics]")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  Precision:   {metrics['precision_weighted']:.4f}")
    print(f"  Recall:      {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score:    {metrics['f1_weighted']:.4f}")

    print(f"\n[Per-Class Metrics]")
    for i, cls_name in enumerate(SPOTTING_CLASSES):
        if i < len(metrics['precision_per_class']):
            print(f"  {cls_name}:")
            print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"    Recall:    {metrics['recall_per_class'][i]:.4f}")
            print(f"    F1:        {metrics['f1_per_class'][i]:.4f}")
            print(f"    Support:   {metrics['support_per_class'][i]}")

    # print(f"\n[Optimal Threshold: {metrics['optimal_threshold']:.4f}]")
    # print(f"  Precision:   {metrics['opt_precision']:.4f}")
    # print(f"  Recall:      {metrics['opt_recall']:.4f}")
    # print(f"  F1 Score:    {metrics['opt_f1']:.4f}")

    print(f"\n[Confusion Matrix]")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"              Normal  Abnormal")
    print(f"  Actual Normal   {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"       Abnormal   {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    print("\n" + "=" * 60)


def compute_video_level_metrics(
    file_scores: Dict[str, Dict],
    agg_method: str = "max",
    threshold: float = 0.5,
) -> Dict:
    """
    Compute video-level (file-level) evaluation metrics.

    Args:
        file_scores: Dictionary from aggregate_by_file().
        agg_method: Aggregation method for video-level score.
            - "max": Max probability across segments (common for anomaly detection)
            - "mean": Mean probability across segments
            - "any": 1 if any segment prob >= threshold, else 0
        threshold: Threshold for binary prediction.

    Returns:
        Dictionary with video-level metrics.
    """
    video_labels = []
    video_probs = []
    video_preds = []

    for npy_path, data in file_scores.items():
        # Ground truth: 1 if any segment is anomaly
        gt_label = data['file_label']
        video_labels.append(gt_label)

        # Aggregated probability
        if agg_method == "max":
            agg_prob = data['max_prob']
        elif agg_method == "mean":
            agg_prob = data['mean_prob']
        elif agg_method == "any":
            agg_prob = 1.0 if data['max_prob'] >= threshold else 0.0
        else:
            agg_prob = data['max_prob']

        video_probs.append(agg_prob)
        video_preds.append(1 if agg_prob >= threshold else 0)

    video_labels = np.array(video_labels)
    video_probs = np.array(video_probs)
    video_preds = np.array(video_preds)

    # Accuracy
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

    # # Optimal threshold via Youden's J
    # if len(np.unique(video_labels)) > 1:
    #     fpr, tpr, thresholds = roc_curve(video_labels, video_probs)
    #     j_scores = tpr - fpr
    #     optimal_idx = np.argmax(j_scores)
    #     optimal_threshold = thresholds[optimal_idx]

    #     optimal_preds = (video_probs >= optimal_threshold).astype(int)
    #     opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
    #         video_labels, optimal_preds, average='binary', zero_division=0
    #     )
    # else:
    #     optimal_threshold = threshold
    #     opt_precision = opt_recall = opt_f1 = float('nan')

    # Per-class breakdown
    n_total = len(video_labels)
    n_anomaly_videos = int(np.sum(video_labels))
    n_normal_videos = n_total - n_anomaly_videos

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
        # 'optimal_threshold': optimal_threshold,
        # 'opt_precision': opt_precision,
        # 'opt_recall': opt_recall,
        # 'opt_f1': opt_f1,
        'n_total': n_total,
        'n_anomaly_videos': n_anomaly_videos,
        'n_normal_videos': n_normal_videos,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'agg_method': agg_method,
        'threshold': threshold,
        'video_labels': video_labels,
        'video_probs': video_probs,
        'video_preds': video_preds,
    }


def print_video_level_report(metrics: Dict):
    """Print video-level evaluation report."""
    print("\n" + "=" * 70)
    print("VIDEO-LEVEL EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nAggregation Method: {metrics['agg_method'].upper()}")
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"Total Videos: {metrics['n_total']} "
          f"(Anomaly: {metrics['n_anomaly_videos']}, Normal: {metrics['n_normal_videos']})")

    print(f"\n[Video-Level Metrics @ threshold={metrics['threshold']:.2f}]")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics['auc_roc']) else "N/A"
    print(f"  AUC-ROC:     {auc_str}")
    prec_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "N/A"
    rec_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "N/A"
    f1_str = f"{metrics['f1']:.4f}" if not np.isnan(metrics['f1']) else "N/A"
    print(f"  Precision:   {prec_str}")
    print(f"  Recall:      {rec_str}")
    print(f"  F1 Score:    {f1_str}")

    # print(f"\n[Optimal Threshold: {metrics['optimal_threshold']:.4f}]")
    # opt_prec_str = f"{metrics['opt_precision']:.4f}" if not np.isnan(metrics['opt_precision']) else "N/A"
    # opt_rec_str = f"{metrics['opt_recall']:.4f}" if not np.isnan(metrics['opt_recall']) else "N/A"
    # opt_f1_str = f"{metrics['opt_f1']:.4f}" if not np.isnan(metrics['opt_f1']) else "N/A"
    # print(f"  Precision:   {opt_prec_str}")
    # print(f"  Recall:      {opt_rec_str}")
    # print(f"  F1 Score:    {opt_f1_str}")

    print(f"\n[Confusion Matrix (Video-Level)]")
    print(f"                Predicted")
    print(f"              Normal  Anomaly")
    print(f"  Actual Normal   {metrics['tn']:5d}   {metrics['fp']:5d}")
    print(f"       Anomaly    {metrics['fn']:5d}   {metrics['tp']:5d}")

    print("\n" + "=" * 70)


def plot_video_level_roc(
    video_metrics: Dict,
    save_path: str,
):
    """Plot and save video-level ROC curve."""
    labels = video_metrics['video_labels']
    probs = video_metrics['video_probs']

    if len(np.unique(labels)) <= 1:
        print("Cannot plot video-level ROC curve: only one class present")
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = video_metrics['auc_roc']

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Video-Level AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Video-Level ROC Curve (Agg: {video_metrics["agg_method"]})', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Video-level ROC curve saved to {save_path}")


def select_files_per_class(
    file_scores: Dict[str, Dict],
    max_files: int,
) -> List[str]:
    """
    Select files for timeline plots ensuring class coverage and diversity.

    Selection strategy:
    1. Group files by class (parent directory name)
    2. From each class, pick:
       a) Best file: highest num_anomaly_segments (or highest max_prob for Normal)
       b) Worst file: lowest segment_acc (most errors, helps diagnose model weaknesses)
    3. Fill remaining slots with highest anomaly segment files across all classes

    Args:
        file_scores: Dictionary mapping npy_path to score info.
        max_files: Maximum number of files to select.

    Returns:
        List of selected npy_paths.
    """
    from collections import defaultdict

    # Group files by class
    class_files: Dict[str, List[str]] = defaultdict(list)
    for npy_path in file_scores:
        class_name = os.path.basename(os.path.dirname(npy_path))
        class_files[class_name].append(npy_path)

    selected = []
    selected_set = set()

    # Step 1: Select best file from each class (ensure class coverage)
    for class_name, paths in sorted(class_files.items()):
        if len(selected) >= max_files:
            break

        if class_name.lower() == 'normal':
            # For Normal class, pick file with highest max_prob (most anomaly-like prediction)
            best_file = max(paths, key=lambda x: file_scores[x]['max_prob'])
        else:
            # For anomaly classes, pick file with most anomaly segments
            best_file = max(paths, key=lambda x: file_scores[x]['num_anomaly_segments'])

        if best_file not in selected_set:
            selected.append((best_file, 'best'))
            selected_set.add(best_file)

    # Step 2: Select worst accuracy file from each class (diagnostic purpose)
    for class_name, paths in sorted(class_files.items()):
        if len(selected) >= max_files:
            break

        # Pick file with lowest segment accuracy
        worst_file = min(paths, key=lambda x: file_scores[x]['segment_acc'])

        if worst_file not in selected_set:
            selected.append((worst_file, 'worst_acc'))
            selected_set.add(worst_file)

    # Step 3: Fill remaining slots with highest anomaly segment files
    if len(selected) < max_files:
        remaining_files = [p for p in file_scores if p not in selected_set]
        sorted_remaining = sorted(
            remaining_files,
            key=lambda x: file_scores[x]['num_anomaly_segments'],
            reverse=True,
        )
        for npy_path in sorted_remaining:
            if len(selected) >= max_files:
                break
            selected.append((npy_path, 'extra'))
            selected_set.add(npy_path)

    # Print selection summary
    print(f"\nTimeline plot file selection:")
    for npy_path, selection_type in selected:
        class_name = os.path.basename(os.path.dirname(npy_path))
        file_stem = os.path.splitext(os.path.basename(npy_path))[0]
        n_anom = file_scores[npy_path]['num_anomaly_segments']
        seg_acc = file_scores[npy_path]['segment_acc']
        tag = f"[{selection_type}]" if selection_type != 'extra' else ""
        print(f"  [{class_name}] {file_stem}: {n_anom} anomaly segs, "
              f"seg_acc={seg_acc:.3f} {tag}")

    # Return only paths (without selection type)
    return [npy_path for npy_path, _ in selected]


def expand_segment_to_frames(
    file_scores: Dict[str, Dict],
    dataset: NPYFeatureDataset,
    fps: float = DEFAULT_FPS,
) -> Dict[str, Dict]:
    """
    Expand segment-level predictions to frame-level predictions.

    Sliding window with special handling for first segment (Window 0):

    Window 0 (first segment, for low latency):
    - Model input:  0-1s
    - Prediction assigned to: 0-1s (frames 0-29)

    Window 1+ (subsequent segments, standard 1-second stride):
    - Window 1: Model input 0-2s → prediction to 1-2s (frames 30-59)
    - Window 2: Model input 1-3s → prediction to 2-3s (frames 60-89)
    - Window N: Model input (N-2)~Ns → prediction to (N-1)~Ns

    This matches the temporal annotation format where predictions are aligned
    with specific 1-second frame intervals (30fps basis).

    Args:
        file_scores: Dictionary from aggregate_by_file() with segment info.
        dataset: NPYFeatureDataset instance (to get sample info).
        fps: Frames per second (default: 30).

    Returns:
        Dictionary mapping npy_path to frame-level data:
            - 'frame_probs': array of frame-level predictions
            - 'frame_labels': array of frame-level ground truth labels
            - 'total_frames': total number of frames in video
            - 'valid_frames': indices of frames with valid predictions
    """
    # Build index: npy_path -> list of segment indices
    npy_to_segment_indices: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset.get_sample_info(idx)
        npy_path = sample['npy_path']
        npy_to_segment_indices[npy_path].append(idx)

    frame_expanded_data = {}

    for npy_path, seg_data in file_scores.items():
        probs_arr = np.array(seg_data['probs'])
        labels_arr = np.array(seg_data['labels'])
        timestamps = np.array(seg_data['timestamps'])  # start_idx for each segment

        # Get segment indices for this file
        if npy_path not in npy_to_segment_indices:
            continue

        segment_indices = npy_to_segment_indices[npy_path]
        if len(segment_indices) != len(probs_arr):
            # Mismatch between segment count; skip this file
            continue

        # Compute total frames from the feature file
        # Each row represents 1 second, so total_frames = num_rows * fps
        feat = np.load(npy_path, mmap_mode='r')
        total_frames = feat.shape[0] * int(fps)

        # Initialize frame-level arrays
        frame_probs = np.zeros(total_frames)
        frame_labels = np.zeros(total_frames, dtype=int)
        frame_count = np.zeros(total_frames)  # For averaging overlapping predictions

        # For each segment, expand prediction to frames
        for seg_order, (dataset_idx, prob, label) in enumerate(
            zip(segment_indices, probs_arr, labels_arr)
        ):
            sample = dataset.get_sample_info(dataset_idx)
            seg_start_time = sample['seg_start_time']  # seconds
            seg_end_time = sample['seg_end_time']      # seconds
            start_idx = sample['start_idx']

            # Window 0 (first segment): special case for low latency
            # - Input:  0-1s, Output: 0-1s (frames 0-29)
            # Remaining windows: standard sliding window with 1-second stride
            # - Input:  (r-1)~(r+1)s, Output: last 1s only
            # - Window 1: Input 0-2s → Output 1-2s (frames 30-59)
            # - Window 2: Input 1-3s → Output 2-3s (frames 60-89)
            if start_idx == 0:
                # Window 0: Assign prediction to first 1 second
                output_start = seg_start_time
                output_end = min(seg_start_time + 1.0, seg_end_time)
            else:
                # Window 1+: Assign prediction to last 1 second of the window
                output_start = seg_end_time - 1.0
                output_end = seg_end_time

            # Convert time to frame indices
            # Frame indexing: frame_idx = time_in_seconds * fps
            # Example (30fps): 0-1 seconds = frames 0-29, 1-2 seconds = frames 30-59
            frame_start = int(output_start * fps)
            frame_end = int(np.ceil(output_end * fps))

            # Clip to valid frame range
            frame_start = max(0, frame_start)
            frame_end = min(total_frames, frame_end)

            if frame_start < frame_end:
                frame_probs[frame_start:frame_end] += prob
                frame_labels[frame_start:frame_end] = label  # Take label from segment
                frame_count[frame_start:frame_end] += 1

        # Average overlapping predictions
        valid_frames = frame_count > 0
        frame_probs[valid_frames] /= frame_count[valid_frames]

        frame_expanded_data[npy_path] = {
            'frame_probs': frame_probs,
            'frame_labels': frame_labels,
            'total_frames': total_frames,
            'valid_frames': np.where(valid_frames)[0],
        }

    return frame_expanded_data


def compute_frame_level_auc(
    frame_expanded_data: Dict[str, Dict],
) -> Dict:
    """
    Compute frame-level AUC metrics across all videos.

    Args:
        frame_expanded_data: Output from expand_segment_to_frames().

    Returns:
        Dictionary with frame-level metrics.
    """
    all_frame_probs = []
    all_frame_labels = []

    for npy_path, data in frame_expanded_data.items():
        # Only include frames that were actually predicted
        valid_mask = np.isin(np.arange(len(data['frame_probs'])), data['valid_frames'])
        if np.any(valid_mask):
            all_frame_probs.extend(data['frame_probs'][valid_mask])
            all_frame_labels.extend(data['frame_labels'][valid_mask])

    all_frame_probs = np.array(all_frame_probs)
    all_frame_labels = np.array(all_frame_labels)

    # Compute AUC
    if len(np.unique(all_frame_labels)) > 1:
        frame_auc = roc_auc_score(all_frame_labels, all_frame_probs)
    else:
        frame_auc = float('nan')

    # Binary predictions at threshold 0.5
    frame_preds = (all_frame_probs >= 0.5).astype(int)
    frame_accuracy = np.mean(frame_preds == all_frame_labels)

    # Precision, Recall, F1
    if len(np.unique(all_frame_labels)) > 1 and len(np.unique(frame_preds)) > 1:
        frame_precision, frame_recall, frame_f1, _ = precision_recall_fscore_support(
            all_frame_labels, frame_preds, average='binary', zero_division=0
        )
    else:
        frame_precision = frame_recall = frame_f1 = float('nan')

    return {
        'auc': frame_auc,
        'accuracy': frame_accuracy,
        'precision': frame_precision,
        'recall': frame_recall,
        'f1': frame_f1,
        'n_frames': len(all_frame_labels),
        'n_anomaly_frames': int(np.sum(all_frame_labels)),
        'n_normal_frames': len(all_frame_labels) - int(np.sum(all_frame_labels)),
    }


def compute_frame_level_ano_auc(
    frame_expanded_data: Dict[str, Dict],
    file_scores: Dict[str, Dict],
) -> Dict:
    """
    Compute frame-level Ano-AUC (anomaly-only AUC) at both segment and frame levels.

    Filters to frames from videos where file_label == 1 (anomaly-containing videos only).

    Args:
        frame_expanded_data: Output from expand_segment_to_frames().
        file_scores: Output from aggregate_by_file().

    Returns:
        Dictionary with frame-level Ano-AUC metrics.
    """
    all_frame_probs = []
    all_frame_labels = []
    per_video_auc = []

    for npy_path, frame_data in frame_expanded_data.items():
        # Only include frames from anomaly videos
        if npy_path not in file_scores or file_scores[npy_path]['file_label'] != 1:
            continue

        valid_mask = np.isin(np.arange(len(frame_data['frame_probs'])), frame_data['valid_frames'])
        if not np.any(valid_mask):
            continue

        frame_probs = frame_data['frame_probs'][valid_mask]
        frame_labels = frame_data['frame_labels'][valid_mask]

        all_frame_probs.extend(frame_probs)
        all_frame_labels.extend(frame_labels)

        # Per-video frame-level AUC
        if len(np.unique(frame_labels)) > 1:
            vid_auc = roc_auc_score(frame_labels, frame_probs)
        else:
            vid_auc = float('nan')

        per_video_auc.append({
            'npy_path': npy_path,
            'auc': vid_auc,
            'n_frames': len(frame_probs),
            'n_anomaly_frames': int(np.sum(frame_labels)),
            'n_normal_frames': len(frame_probs) - int(np.sum(frame_labels)),
        })

    all_frame_probs = np.array(all_frame_probs)
    all_frame_labels = np.array(all_frame_labels)

    n_anomaly_videos = len(per_video_auc)
    n_total_frames = len(all_frame_labels)
    n_anomaly_frames = int(np.sum(all_frame_labels))
    n_normal_frames = n_total_frames - n_anomaly_frames

    # --- Frame-level Ano-AUC (pooled) ---
    if len(np.unique(all_frame_labels)) > 1:
        frame_ano_auc = roc_auc_score(all_frame_labels, all_frame_probs)
    else:
        frame_ano_auc = float('nan')

    # Default threshold metrics (frame-level)
    frame_preds_05 = (all_frame_probs >= 0.5).astype(int)
    if len(np.unique(all_frame_labels)) > 1:
        frame_prec_05, frame_rec_05, frame_f1_05, _ = precision_recall_fscore_support(
            all_frame_labels, frame_preds_05, average='binary', zero_division=0
        )
    else:
        frame_prec_05 = frame_rec_05 = frame_f1_05 = float('nan')

    # --- Video-level Ano-AUC (macro-average of per-video AUCs) ---
    valid_aucs = [v['auc'] for v in per_video_auc if not np.isnan(v['auc'])]
    vid_ano_auc_macro = np.mean(valid_aucs) if valid_aucs else float('nan')
    vid_ano_auc_std = np.std(valid_aucs) if valid_aucs else float('nan')
    n_valid_videos = len(valid_aucs)
    n_skipped_videos = n_anomaly_videos - n_valid_videos

    return {
        # Frame-level
        'frame_ano_auc': frame_ano_auc,
        'frame_precision_05': frame_prec_05,
        'frame_recall_05': frame_rec_05,
        'frame_f1_05': frame_f1_05,
        # Video-level (per-video frame AUC)
        'vid_ano_auc_macro': vid_ano_auc_macro,
        'vid_ano_auc_std': vid_ano_auc_std,
        'n_valid_videos': n_valid_videos,
        'n_skipped_videos': n_skipped_videos,
        # Counts
        'n_anomaly_videos': n_anomaly_videos,
        'n_total_frames': n_total_frames,
        'n_anomaly_frames': n_anomaly_frames,
        'n_normal_frames': n_normal_frames,
        # Per-video breakdown
        'per_video_auc': per_video_auc,
    }


def print_frame_level_report(frame_metrics: Dict):
    """Print frame-level evaluation report."""
    print("\n" + "=" * 70)
    print("FRAME-LEVEL EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n[Frame-Level Metrics @ threshold=0.5]")
    print(f"  Total frames: {frame_metrics['n_frames']}")
    print(f"    Normal frames:   {frame_metrics['n_normal_frames']}")
    print(f"    Anomaly frames:  {frame_metrics['n_anomaly_frames']}")

    auc_str = f"{frame_metrics['auc']:.4f}" if not np.isnan(frame_metrics['auc']) else "N/A"
    print(f"  AUC-ROC:     {auc_str}")
    print(f"  Accuracy:    {frame_metrics['accuracy']:.4f}")

    prec_str = f"{frame_metrics['precision']:.4f}" if not np.isnan(frame_metrics['precision']) else "N/A"
    rec_str = f"{frame_metrics['recall']:.4f}" if not np.isnan(frame_metrics['recall']) else "N/A"
    f1_str = f"{frame_metrics['f1']:.4f}" if not np.isnan(frame_metrics['f1']) else "N/A"
    print(f"  Precision:   {prec_str}")
    print(f"  Recall:      {rec_str}")
    print(f"  F1 Score:    {f1_str}")

    print("\n" + "=" * 70)


def print_unified_evaluation_report(
    metrics: Dict,
    ano_metrics: Dict,
    video_metrics: Dict,
    frame_metrics: Dict,
    frame_ano_metrics: Dict,
    threshold: float = 0.5,
):
    """Print unified evaluation report combining all metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"@ threshold={threshold:.2f}\n")

    # Segment-level metrics
    print("[Segment-Level Metrics]")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics['auc_roc']) else "N/A"
    print(f"  AUC-ROC:     {auc_str}")
    ano_auc_str = f"{ano_metrics['seg_ano_auc']:.4f}" if not np.isnan(ano_metrics['seg_ano_auc']) else "N/A"
    print(f"  Ano-AUC:     {ano_auc_str} (Scope: {ano_metrics['n_anomaly_videos']} anomaly videos, "
          f"{ano_metrics['n_total_segments']} segments)")

    # Video-level metrics
    print("\n[Video-Level Metrics]")
    print(f"  Accuracy:    {video_metrics['accuracy']:.4f}")
    vid_auc_str = f"{video_metrics['auc_roc']:.4f}" if not np.isnan(video_metrics['auc_roc']) else "N/A"
    print(f"  AUC-ROC:     {vid_auc_str}")
    vid_ano_auc_str = f"{video_metrics['auc_roc']:.4f}" if not np.isnan(video_metrics['auc_roc']) else "N/A"
    # Note: Using video-level AUC as Ano-AUC proxy
    print(f"  Ano-AUC:     {ano_metrics['vid_ano_auc_macro']:.4f} (+/- {ano_metrics['vid_ano_auc_std']:.4f})")

    # Frame-level metrics
    print("\n[Frame-Level Metrics]")
    print(f"  Total frames: {frame_metrics['n_frames']}")
    print(f"    Normal frames:   {frame_metrics['n_normal_frames']}")
    print(f"    Anomaly frames:  {frame_metrics['n_anomaly_frames']}")
    print()
    frame_auc_str = f"{frame_metrics['auc']:.4f}" if not np.isnan(frame_metrics['auc']) else "N/A"
    print(f"  AUC-ROC:     {frame_auc_str}")
    print(f"  Accuracy:    {frame_metrics['accuracy']:.4f}")
    frame_ano_auc_str = f"{frame_ano_metrics['frame_ano_auc']:.4f}" if not np.isnan(frame_ano_metrics['frame_ano_auc']) else "N/A"
    print(f"  Ano-AUC:     {frame_ano_auc_str} "
          f"(Scope: {frame_ano_metrics['n_anomaly_videos']} anomaly videos, "
          f"{frame_ano_metrics['n_total_frames']} total frames "
          f"(anomaly: {frame_ano_metrics['n_anomaly_frames']}, "
          f"normal: {frame_ano_metrics['n_normal_frames']}))")

    print("\n" + "=" * 60)


def print_frame_level_ano_report(frame_ano_metrics: Dict, top_k: int = 0):
    """Print frame-level Ano-AUC evaluation report."""
    print("\n" + "=" * 70)
    print("FRAME-LEVEL ANOMALY-ONLY EVALUATION (Frame Ano-AUC)")
    print("=" * 70)

    print(f"\nScope: {frame_ano_metrics['n_anomaly_videos']} anomaly videos, "
          f"{frame_ano_metrics['n_total_frames']} total frames "
          f"(anomaly: {frame_ano_metrics['n_anomaly_frames']}, "
          f"normal: {frame_ano_metrics['n_normal_frames']})")

    # --- Frame-level ---
    print(f"\n[Frame-Level Ano-AUC (pooled)]")
    auc_str = f"{frame_ano_metrics['frame_ano_auc']:.4f}" if not np.isnan(frame_ano_metrics['frame_ano_auc']) else "N/A"
    print(f"  Ano-AUC:     {auc_str}")

    print(f"  @ threshold=0.5:")
    print(f"    Precision: {frame_ano_metrics['frame_precision_05']:.4f}")
    print(f"    Recall:    {frame_ano_metrics['frame_recall_05']:.4f}")
    print(f"    F1:        {frame_ano_metrics['frame_f1_05']:.4f}")

    # --- Video-level (per-video frame AUC) ---
    print(f"\n[Per-Video Frame-Level Ano-AUC (macro-avg)]")
    vid_auc_str = (f"{frame_ano_metrics['vid_ano_auc_macro']:.4f}"
                   if not np.isnan(frame_ano_metrics['vid_ano_auc_macro']) else "N/A")
    vid_std_str = (f"{frame_ano_metrics['vid_ano_auc_std']:.4f}"
                   if not np.isnan(frame_ano_metrics['vid_ano_auc_std']) else "N/A")
    print(f"  Ano-AUC:     {vid_auc_str} (+/- {vid_std_str})")
    print(f"  Valid videos: {frame_ano_metrics['n_valid_videos']} / {frame_ano_metrics['n_anomaly_videos']}"
          f"  (skipped {frame_ano_metrics['n_skipped_videos']} single-class videos)")

    # --- Top-k per-video AUC ---
    if top_k > 0:
        per_video = frame_ano_metrics['per_video_auc']
        # Sort by AUC descending (NaN goes last)
        ranked = sorted(per_video, key=lambda v: v['auc'] if not np.isnan(v['auc']) else -1, reverse=True)
        show = ranked[:top_k]

        print(f"\n  Top-{top_k} videos by per-video frame Ano-AUC:")
        print(f"  {'Rank':<5} {'Class':<15} {'File':<35} {'AUC':>7} "
              f"{'Frames':>8} {'Anom':>8}")
        print("  " + "-" * 85)
        for rank, v in enumerate(show, 1):
            class_name = os.path.basename(os.path.dirname(v['npy_path']))
            file_stem = os.path.splitext(os.path.basename(v['npy_path']))[0]
            auc_s = f"{v['auc']:.4f}" if not np.isnan(v['auc']) else "N/A"
            print(f"  {rank:<5} {class_name:<15} {file_stem:<35} "
                  f"{auc_s:>7} {v['n_frames']:>8} {v['n_anomaly_frames']:>8}")
        print("  " + "-" * 85)

    print("\n" + "=" * 70)


def compute_anomaly_only_auc(
    file_scores: Dict[str, Dict],
) -> Dict:
    """
    Compute Ano-AUC at both segment-level and video-level.

    Filters to videos where file_label == 1 (anomaly-containing videos only).

    Segment-level Ano-AUC:
        Pool all segments from anomaly videos and compute a single AUC.
        Measures overall temporal localization across all anomaly videos.

    Video-level Ano-AUC:
        Compute per-video AUC (requires both normal and anomaly segments
        within a single video), then macro-average across videos.
        Measures per-video temporal localization quality.

    Returns:
        Dictionary with segment/video-level Ano-AUC, per-video AUC list, etc.
    """
    all_probs = []
    all_labels = []
    per_video_auc = []  # (npy_path, auc, n_segments, n_anomaly, n_normal, f1)

    for npy_path, data in file_scores.items():
        if data['file_label'] != 1:
            continue

        probs_arr = np.array(data['probs'])
        labels_arr = np.array(data['labels'])
        all_probs.extend(probs_arr)
        all_labels.extend(labels_arr)

        # Per-video AUC (only computable if video has both classes)
        if len(np.unique(labels_arr)) > 1:
            vid_auc = roc_auc_score(labels_arr, probs_arr)
        else:
            vid_auc = float('nan')

        per_video_auc.append({
            'npy_path': npy_path,
            'auc': vid_auc,
            'n_segments': len(probs_arr),
            'n_anomaly': int(np.sum(labels_arr)),
            'n_normal': int(np.sum(labels_arr == 0)),
            'segment_f1': data['segment_f1'],
            'segment_acc': data['segment_acc'],
        })

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    n_anomaly_videos = len(per_video_auc)
    n_total_segments = len(all_labels)
    n_anomaly_segments = int(np.sum(all_labels))
    n_normal_segments = n_total_segments - n_anomaly_segments

    # --- Segment-level Ano-AUC (pooled) ---
    if len(np.unique(all_labels)) > 1:
        seg_ano_auc = roc_auc_score(all_labels, all_probs)
    else:
        seg_ano_auc = float('nan')

    # # Optimal threshold within anomaly videos (segment-level)
    # if len(np.unique(all_labels)) > 1:
    #     fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    #     j_scores = tpr - fpr
    #     optimal_idx = np.argmax(j_scores)
    #     optimal_threshold = thresholds[optimal_idx]

    #     optimal_preds = (all_probs >= optimal_threshold).astype(int)
    #     opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
    #         all_labels, optimal_preds, average='binary', zero_division=0
    #     )
    # else:
    #     optimal_threshold = 0.5
    #     opt_precision = opt_recall = opt_f1 = float('nan')

    # Default threshold metrics (segment-level)
    preds_05 = (all_probs >= 0.5).astype(int)
    if len(np.unique(all_labels)) > 1:
        prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(
            all_labels, preds_05, average='binary', zero_division=0
        )
    else:
        prec_05 = rec_05 = f1_05 = float('nan')

    # --- Video-level Ano-AUC (macro-average of per-video AUCs) ---
    valid_aucs = [v['auc'] for v in per_video_auc if not np.isnan(v['auc'])]
    vid_ano_auc_macro = np.mean(valid_aucs) if valid_aucs else float('nan')
    vid_ano_auc_std = np.std(valid_aucs) if valid_aucs else float('nan')
    n_valid_videos = len(valid_aucs)
    n_skipped_videos = n_anomaly_videos - n_valid_videos

    return {
        # Segment-level
        'seg_ano_auc': seg_ano_auc,
        # 'optimal_threshold': optimal_threshold,
        # 'opt_precision': opt_precision,
        # 'opt_recall': opt_recall,
        # 'opt_f1': opt_f1,
        'precision_05': prec_05,
        'recall_05': rec_05,
        'f1_05': f1_05,
        # Video-level
        'vid_ano_auc_macro': vid_ano_auc_macro,
        'vid_ano_auc_std': vid_ano_auc_std,
        'n_valid_videos': n_valid_videos,
        'n_skipped_videos': n_skipped_videos,
        # Counts
        'n_anomaly_videos': n_anomaly_videos,
        'n_total_segments': n_total_segments,
        'n_anomaly_segments': n_anomaly_segments,
        'n_normal_segments': n_normal_segments,
        # Per-video breakdown
        'per_video_auc': per_video_auc,
    }


def print_anomaly_only_report(ano_metrics: Dict, top_k: int = 0):
    """Print Ano-AUC evaluation report (segment + video level)."""
    print("\n" + "=" * 70)
    print("ANOMALY-ONLY EVALUATION (Ano-AUC)")
    print("=" * 70)

    print(f"\nScope: {ano_metrics['n_anomaly_videos']} anomaly videos, "
          f"{ano_metrics['n_total_segments']} total segments "
          f"(anomaly: {ano_metrics['n_anomaly_segments']}, "
          f"normal: {ano_metrics['n_normal_segments']})")

    # --- Segment-level ---
    print(f"\n[Segment-Level Ano-AUC (pooled)]")
    auc_str = f"{ano_metrics['seg_ano_auc']:.4f}" if not np.isnan(ano_metrics['seg_ano_auc']) else "N/A"
    print(f"  Ano-AUC:     {auc_str}")

    print(f"  @ threshold=0.5:")
    print(f"    Precision: {ano_metrics['precision_05']:.4f}")
    print(f"    Recall:    {ano_metrics['recall_05']:.4f}")
    print(f"    F1:        {ano_metrics['f1_05']:.4f}")

    # print(f"  @ optimal threshold={ano_metrics['optimal_threshold']:.4f}:")
    # opt_p = f"{ano_metrics['opt_precision']:.4f}" if not np.isnan(ano_metrics['opt_precision']) else "N/A"
    # opt_r = f"{ano_metrics['opt_recall']:.4f}" if not np.isnan(ano_metrics['opt_recall']) else "N/A"
    # opt_f = f"{ano_metrics['opt_f1']:.4f}" if not np.isnan(ano_metrics['opt_f1']) else "N/A"
    # print(f"    Precision: {opt_p}")
    # print(f"    Recall:    {opt_r}")
    # print(f"    F1:        {opt_f}")

    # --- Video-level ---
    print(f"\n[Video-Level Ano-AUC (macro-avg of per-video AUC)]")
    vid_auc_str = (f"{ano_metrics['vid_ano_auc_macro']:.4f}"
                   if not np.isnan(ano_metrics['vid_ano_auc_macro']) else "N/A")
    vid_std_str = (f"{ano_metrics['vid_ano_auc_std']:.4f}"
                   if not np.isnan(ano_metrics['vid_ano_auc_std']) else "N/A")
    print(f"  Ano-AUC:     {vid_auc_str} (+/- {vid_std_str})")
    print(f"  Valid videos: {ano_metrics['n_valid_videos']} / {ano_metrics['n_anomaly_videos']}"
          f"  (skipped {ano_metrics['n_skipped_videos']} single-class videos)")

    # --- Top-k per-video AUC ---
    if top_k > 0:
        per_video = ano_metrics['per_video_auc']
        # Sort by AUC descending (NaN goes last)
        ranked = sorted(per_video, key=lambda v: v['auc'] if not np.isnan(v['auc']) else -1, reverse=True)
        show = ranked[:top_k]

        print(f"\n  Top-{top_k} videos by per-video Ano-AUC:")
        print(f"  {'Rank':<5} {'Class':<15} {'File':<35} {'AUC':>7} {'F1':>7} "
              f"{'Acc':>7} {'Segs':>5} {'Anom':>5}")
        print("  " + "-" * 95)
        for rank, v in enumerate(show, 1):
            class_name = os.path.basename(os.path.dirname(v['npy_path']))
            file_stem = os.path.splitext(os.path.basename(v['npy_path']))[0]
            auc_s = f"{v['auc']:.4f}" if not np.isnan(v['auc']) else "N/A"
            print(f"  {rank:<5} {class_name:<15} {file_stem:<35} "
                  f"{auc_s:>7} {v['segment_f1']:>7.4f} "
                  f"{v['segment_acc']:>7.4f} {v['n_segments']:>5} {v['n_anomaly']:>5}")
        print("  " + "-" * 95)

    print("\n" + "=" * 70)


def select_top_segment_f1_files(
    file_scores: Dict[str, Dict],
    top_n: int = 5,
) -> List[str]:
    """
    Select top N files by segment-level F1 score.

    Only considers files that contain anomaly segments (file_label == 1),
    since F1 is 0.0 for purely normal files by definition.

    Args:
        file_scores: Dictionary from aggregate_by_file().
        top_n: Number of top files to select.

    Returns:
        List of selected npy_paths sorted by segment F1 (descending).
    """
    # Filter to files with anomaly ground truth (F1 is meaningful)
    anomaly_files = [
        (path, data) for path, data in file_scores.items()
        if data['file_label'] == 1
    ]

    # Sort by segment F1 descending
    anomaly_files.sort(key=lambda x: x[1]['segment_f1'], reverse=True)

    selected = anomaly_files[:top_n]

    print(f"\nTop-{top_n} files by segment-level F1 score:")
    print(f"{'Rank':<5} {'Class':<15} {'File':<35} {'F1':>7} {'Prec':>7} "
          f"{'Rec':>7} {'Acc':>7} {'Segs':>5} {'Anom':>5}")
    print("-" * 100)
    for rank, (npy_path, data) in enumerate(selected, 1):
        class_name = os.path.basename(os.path.dirname(npy_path))
        file_stem = os.path.splitext(os.path.basename(npy_path))[0]
        print(f"{rank:<5} {class_name:<15} {file_stem:<35} "
              f"{data['segment_f1']:>7.4f} {data['segment_precision']:>7.4f} "
              f"{data['segment_recall']:>7.4f} {data['segment_acc']:>7.4f} "
              f"{data['num_segments']:>5} {data['num_anomaly_segments']:>5}")
    print("-" * 100)

    return [path for path, _ in selected]


def evaluate(
    checkpoint_path: str,
    feature_dir: str,
    annotation_path: str,
    dataset_name: str,
    embed_dim: int = 512,
    unit_duration: int = 2,
    overlap_ratio: float = 0.5,
    temporal_agg: str = "mean",
    dropout_rate: float = 0.5,
    model_type: str = "linear",
    batch_size: int = 64,
    num_workers: int = 4,
    output_dir: str = "./evaluation",
    plot_files: int = 10,
    top_f1_files: int = 0,
    top_ano_auc_files: int = 0,
    seed: int = 42,
):
    """Main evaluation function."""
    set_seed(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset (test split)
    print("\nLoading test dataset...")
    dataset = NPYFeatureDataset(
        feature_dir=feature_dir,
        annotation_path=annotation_path,
        unit_duration=unit_duration,
        overlap_ratio=overlap_ratio,
        strict_normal_sampling=False,  # Keep all windows for evaluation
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

    # Create and load model
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

    print(f"\nLoading checkpoint from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model=model, device=device)
    model = model.to(device)

    save_suffix = dataset_name + "_" + model_type
    # output_dir = os.path.join(output_dir, save_suffix)
    output_dir = "./evaluation_" + save_suffix
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(model, data_loader, device, model_type=model_type)

    # Compute metrics
    metrics = compute_metrics(results['labels'], results['probs'], results['preds'])

    # Compute per-class metrics
    class_metrics = compute_per_class_metrics(
        dataset, results['preds'], results['labels'], results['probs']
    )

    # Plot curves
    plot_roc_curve(
        results['labels'], results['probs'],
        os.path.join(output_dir, "roc_curve.png"),
    )
    plot_precision_recall_curve(
        results['labels'], results['probs'],
        os.path.join(output_dir, "pr_curve.png"),
    )
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Aggregate by file and plot timelines
    file_scores = aggregate_by_file(
        dataset, results['probs'], results['labels'],
        results['logits_normal'], results['logits_anomaly'],
    )

    # Video-level evaluation
    video_metrics = compute_video_level_metrics(file_scores, agg_method="mean") # mean or max

    # Anomaly-only AUC (segment-level + video-level)
    ano_metrics = compute_anomaly_only_auc(file_scores)

    # Frame-level evaluation
    print("\nExpanding segment-level predictions to frame-level...")
    frame_expanded_data = expand_segment_to_frames(file_scores, dataset, fps=dataset.fps)

    frame_metrics = compute_frame_level_auc(frame_expanded_data)

    # Frame-level Ano-AUC
    frame_ano_metrics = compute_frame_level_ano_auc(frame_expanded_data, file_scores)

    # Print unified evaluation report
    print_unified_evaluation_report(
        metrics, ano_metrics, video_metrics, frame_metrics, frame_ano_metrics
    )

    # Print detailed reports (optional)
    print_metrics_report(metrics)
    print_per_class_report(class_metrics)
    print_video_level_report(video_metrics)
    print_anomaly_only_report(ano_metrics, top_k=top_ano_auc_files)
    print_frame_level_ano_report(frame_ano_metrics, top_k=top_ano_auc_files)

    # Plot video-level ROC curve
    plot_video_level_roc(
        video_metrics,
        os.path.join(output_dir, "video_level_roc_curve.png"),
    )

    if plot_files > 0:
        # Select files ensuring at least 1 per class
        files_to_plot = select_files_per_class(file_scores, plot_files)
        print(f"\nSelected {len(files_to_plot)} files for timeline plots "
              f"(ensuring class coverage)")
        for npy_path in files_to_plot:
            plot_file_timeline(
                file_scores, npy_path,
                os.path.join(output_dir, "timeline_plots"),
            )

    # Top Ano-AUC timeline plots
    if top_ano_auc_files > 0:
        ranked = sorted(
            ano_metrics['per_video_auc'],
            key=lambda v: v['auc'] if not np.isnan(v['auc']) else -1,
            reverse=True,
        )
        top_auc_paths = [v['npy_path'] for v in ranked[:top_ano_auc_files]]
        print(f"\nPlotting timeline for top-{top_ano_auc_files} Ano-AUC files...")
        for npy_path in top_auc_paths:
            plot_file_timeline(
                file_scores, npy_path,
                os.path.join(output_dir, "timeline_top_ano_auc"),
            )

    # Top segment-level F1 files
    if top_f1_files > 0:
        top_f1_paths = select_top_segment_f1_files(file_scores, top_n=top_f1_files)
        print(f"\nPlotting timeline for top-{top_f1_files} segment F1 files...")
        for npy_path in top_f1_paths:
            plot_file_timeline(
                file_scores, npy_path,
                os.path.join(output_dir, "timeline_top_f1"),
            )

    # Save results
    # np.savez(
    #     os.path.join(output_dir, "results.npz"),
    #     # Segment-level
    #     labels=results['labels'],
    #     probs=results['probs'],
    #     logits_normal=results['logits_normal'],
    #     logits_anomaly=results['logits_anomaly'],
    #     preds=results['preds'],
    #     # Video-level
    #     video_labels=video_metrics['video_labels'],
    #     video_probs=video_metrics['video_probs'],
    #     video_preds=video_metrics['video_preds'],
    # )
    # print(f"\nResults saved to {output_dir}")

    return metrics, file_scores, video_metrics, frame_metrics, frame_ano_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FeatureSpottingModel on pre-extracted npy features "
    )

    # Data
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Path to test feature directory (e.g., .../MCi2-S0-6/test)")
    # parser.add_argument("--annotation-path", type=str, required=True,
    #                     help="Path to Text file or annotation directory with *_timestamp.csv files")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Name of the dataset (ETRI or UCF) for identifying the source dataset")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["linear", "resmlp"],
                        help="Model type: linear (FeatureSpottingModel) or resmlp (ResidualMLPSpottingModel)")
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

    # Evaluation
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    # parser.add_argument("--output-dir", type=str, default="./evaluation_etri",
    #                     help="Directory to save outputs")
    parser.add_argument("--plot-files", type=int, default=3,
                        help="Number of files to plot timelines for")
    parser.add_argument("--top-f1-files", type=int, default=5,
                        help="Number of top segment-level F1 files to plot timelines for")
    parser.add_argument("--top-ano-auc-files", type=int, default=5,
                        help="Number of top per-video Ano-AUC files to show and plot timelines for")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    annotation_path = "./annotation/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
    args = parse_args()

    print("=" * 60)
    print("Feature-based Anomaly Action Spotting - Evaluation")
    print("=" * 60)
    print(f"Feature dir: {args.feature_dir}")
    print(f"Annotation path: {annotation_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Temporal agg: {args.temporal_agg}")
    print(f"Unit duration: {args.unit_duration}s")
    print(f"Overlap ratio: {args.overlap_ratio}")
    print("=" * 60)

    evaluate(
        checkpoint_path=args.checkpoint,
        feature_dir=args.feature_dir,
        annotation_path=annotation_path,
        dataset_name=args.dataset_name,
        embed_dim=args.embed_dim,
        unit_duration=args.unit_duration,
        overlap_ratio=args.overlap_ratio,
        temporal_agg=args.temporal_agg,
        dropout_rate=args.dropout,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
#        output_dir=args.output_dir,
        plot_files=args.plot_files,
        top_f1_files=args.top_f1_files,
        top_ano_auc_files=args.top_ano_auc_files,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
