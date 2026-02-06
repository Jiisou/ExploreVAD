#!/usr/bin/env python3
"""
Evaluation script for pre-extracted npy features (ETRI dataset).

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
from dataset import ETRIFeatureDataset
from model import FeatureSpottingModel, ResidualMLPSpottingModel
from utils import set_seed, get_device, load_checkpoint
from config import SPOTTING_CLASSES


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

    # Optimal threshold via Youden's J statistic
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

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
        'optimal_threshold': optimal_threshold,
        'opt_precision': opt_precision,
        'opt_recall': opt_recall,
        'opt_f1': opt_f1,
    }


def compute_per_class_metrics(
    dataset: ETRIFeatureDataset,
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Dict]:
    """
    Compute binary prediction metrics per anomaly class.

    Groups samples by parent directory name (class) and computes
    accuracy, precision, recall, F1 for each class.

    Args:
        dataset: ETRIFeatureDataset instance.
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
    dataset: ETRIFeatureDataset,
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
        file_scores[npy_path]['timestamps'].append(sample['start_sec'])

    for npy_path in file_scores:
        probs_arr = np.array(file_scores[npy_path]['probs'])
        labels_arr = np.array(file_scores[npy_path]['labels'])

        # Calculate segment-level accuracy (threshold at 0.5)
        preds_arr = (probs_arr >= 0.5).astype(int)
        segment_acc = np.mean(preds_arr == labels_arr)

        file_scores[npy_path].update({
            'max_prob': np.max(probs_arr),
            'mean_prob': np.mean(probs_arr),
            'num_segments': len(probs_arr),
            'num_anomaly_segments': int(np.sum(labels_arr)),
            'file_label': 1 if np.any(labels_arr == 1) else 0,
            'segment_acc': segment_acc,
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
    plt.title('ROC Curve - Feature Anomaly Spotting (ETRI)', fontsize=14)
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
    plt.title('Precision-Recall Curve - Feature Anomaly Spotting (ETRI)', fontsize=14)
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
    ax.set_title('Confusion Matrix - Feature Anomaly Spotting (ETRI)', fontsize=14)
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
    print("EVALUATION RESULTS (ETRI Feature)")
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


def evaluate(
    checkpoint_path: str,
    feature_dir: str,
    annotation_dir: str,
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
    seed: int = 42,
):
    """Main evaluation function."""
    set_seed(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset (test split)
    print("\nLoading test dataset...")
    dataset = ETRIFeatureDataset(
        feature_dir=feature_dir,
        annotation_dir=annotation_dir,
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

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(model, data_loader, device, model_type=model_type)

    # Compute metrics
    metrics = compute_metrics(results['labels'], results['probs'], results['preds'])
    print_metrics_report(metrics)

    # Compute per-class metrics
    class_metrics = compute_per_class_metrics(
        dataset, results['preds'], results['labels'], results['probs']
    )
    print_per_class_report(class_metrics)

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
    
    save_suffix = dataset_name + "_" + model_type
    # output_dir = os.path.join(output_dir, save_suffix)
    output_dir = "./evaluation_" + save_suffix
    os.makedirs(output_dir, exist_ok=True)

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

    # Save results
    np.savez(
        os.path.join(output_dir, "results.npz"),
        labels=results['labels'],
        probs=results['probs'],
        logits_normal=results['logits_normal'],
        logits_anomaly=results['logits_anomaly'],
        preds=results['preds'],
    )
    print(f"\nResults saved to {output_dir}")

    return metrics, file_scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FeatureSpottingModel on pre-extracted npy features (ETRI)"
    )

    # Data
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Path to test feature directory (e.g., .../MCi2-S0-6/test)")
    parser.add_argument("--annotation-dir", type=str, required=True,
                        help="Path to annotation directory with *_timestamp.csv files")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Name of the dataset (ETRI or UCF) for identifying the source dataset")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="linear",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Feature-based Anomaly Action Spotting - Evaluation")
    print("=" * 60)
    print(f"Feature dir: {args.feature_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
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
        annotation_dir=args.annotation_dir,
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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
