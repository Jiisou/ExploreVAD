#!/usr/bin/env python3
"""
Inference script for pre-extracted npy features (ETRI dataset).

Uses FeatureSpottingModel (temporal aggregation + classification head only).

Features:
- Process single npy file or directory of npy files
- Sliding window anomaly scoring
- Export results to CSV
- Anomaly timeline visualization
- Contiguous anomaly segment detection
"""

import argparse
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import FeatureSpottingModel, ResidualMLPSpottingModel
from utils import get_device, load_checkpoint


@torch.no_grad()
def process_npy(
    model: torch.nn.Module,
    npy_path: str,
    device: torch.device,
    unit_duration: int = 2,
    overlap_ratio: float = 0.0,
    batch_size: int = 64,
    model_type: str = "linear",
) -> List[Dict]:
    """
    Process a single npy feature file and return anomaly scores per window.

    Args:
        model: Model instance (FeatureSpottingModel or ResidualMLPSpottingModel).
        npy_path: Path to .npy feature file (T_seconds, D).
        device: Torch device.
        unit_duration: Window size in seconds.
        overlap_ratio: Overlap ratio between windows.
        batch_size: Batch size for inference.
        model_type: "linear" or "resmlp".

    Returns:
        List of dicts with window info and anomaly scores.
    """
    feat = np.load(npy_path)  # (T_seconds, D)
    total_seconds = feat.shape[0]

    if total_seconds < unit_duration:
        print(f"Warning: {npy_path} has {total_seconds}s, shorter than unit_duration={unit_duration}s")
        return []

    # Sliding window
    stride = max(1, int(unit_duration * (1.0 - overlap_ratio)))
    num_windows = max(0, (total_seconds - unit_duration) // stride + 1)

    if num_windows == 0:
        return []

    model.eval()
    results = []

    # Collect all windows
    windows = []
    window_meta = []
    for i in range(num_windows):
        start_sec = i * stride
        end_sec = start_sec + unit_duration
        window = feat[start_sec:end_sec]  # (unit_duration, D)
        windows.append(window)
        window_meta.append({'start_sec': start_sec, 'end_sec': end_sec})

    # Batch inference
    windows_tensor = torch.from_numpy(np.array(windows)).float()  # (N, T, D)

    for batch_start in range(0, len(windows_tensor), batch_size):
        batch = windows_tensor[batch_start:batch_start + batch_size].to(device)
        outputs = model(batch)

        if model_type == "resmlp":
            logits = outputs.squeeze(1)  # (B,)
            anomaly_probs = torch.sigmoid(logits).cpu().numpy()
            logits_anomaly = logits.cpu().numpy()
            logits_normal = (-logits).cpu().numpy()
        else:
            probs = torch.softmax(outputs, dim=1)
            anomaly_probs = probs[:, 1].cpu().numpy()
            logits_normal = outputs[:, 0].cpu().numpy()
            logits_anomaly = outputs[:, 1].cpu().numpy()

        for j in range(len(anomaly_probs)):
            idx = batch_start + j
            results.append({
                'start_sec': window_meta[idx]['start_sec'],
                'end_sec': window_meta[idx]['end_sec'],
                'anomaly_score': float(anomaly_probs[j]),
                'logit_normal': float(logits_normal[j]),
                'logit_anomaly': float(logits_anomaly[j]),
            })

    return results


def save_results_csv(
    results: List[Dict],
    output_path: str,
    file_name: Optional[str] = None,
):
    """Save inference results to CSV."""
    df = pd.DataFrame(results)
    if file_name:
        df.insert(0, 'file_name', file_name)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def visualize_results(
    results: List[Dict],
    file_name: str,
    output_path: str,
    threshold: float = 0.5,
):
    """Visualize normal and anomaly logits over time."""
    timestamps = [r['start_sec'] for r in results]
    logits_normal = [r['logit_normal'] for r in results]
    logits_anomaly = [r['logit_anomaly'] for r in results]

    all_logits = logits_normal + logits_anomaly
    y_min, y_max = min(all_logits), max(all_logits)
    y_margin = max(0.5, (y_max - y_min) * 0.1)

    plt.figure(figsize=(14, 5))
    plt.plot(timestamps, logits_normal, 'k-', linewidth=1, alpha=0.7, label='Normal Logit')
    plt.plot(timestamps, logits_anomaly, 'b-', linewidth=1.5, alpha=0.8, label='Anomaly Logit')

    # Mark high-score regions
    scores = [r['anomaly_score'] for r in results]
    high_mask = np.array(scores) >= threshold
    if np.any(high_mask):
        unit_dur = results[0]['end_sec'] - results[0]['start_sec']
        for t, is_high in zip(timestamps, high_mask):
            if is_high:
                plt.axvspan(t, t + unit_dur, alpha=0.15, color='red')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Logit', fontsize=12)
    plt.title(f'Anomaly Inference: {file_name}', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")


def find_anomaly_segments(
    results: List[Dict],
    threshold: float = 0.5,
    min_duration: float = 1.0,
) -> List[Dict]:
    """Find contiguous anomaly segments above threshold."""
    anomalies = []
    current_anomaly = None

    for r in results:
        is_anomaly = r['anomaly_score'] >= threshold

        if is_anomaly:
            if current_anomaly is None:
                current_anomaly = {
                    'start_sec': r['start_sec'],
                    'end_sec': r['end_sec'],
                    'max_score': r['anomaly_score'],
                    'scores': [r['anomaly_score']],
                }
            else:
                current_anomaly['end_sec'] = r['end_sec']
                current_anomaly['max_score'] = max(
                    current_anomaly['max_score'], r['anomaly_score']
                )
                current_anomaly['scores'].append(r['anomaly_score'])
        else:
            if current_anomaly is not None:
                duration = current_anomaly['end_sec'] - current_anomaly['start_sec']
                if duration >= min_duration:
                    current_anomaly['mean_score'] = np.mean(current_anomaly['scores'])
                    current_anomaly['duration'] = duration
                    del current_anomaly['scores']
                    anomalies.append(current_anomaly)
                current_anomaly = None

    # Handle last segment
    if current_anomaly is not None:
        duration = current_anomaly['end_sec'] - current_anomaly['start_sec']
        if duration >= min_duration:
            current_anomaly['mean_score'] = np.mean(current_anomaly['scores'])
            current_anomaly['duration'] = duration
            del current_anomaly['scores']
            anomalies.append(current_anomaly)

    return anomalies


def print_summary(results: List[Dict], npy_path: str, threshold: float = 0.5):
    """Print summary of inference results."""
    scores = [r['anomaly_score'] for r in results]
    total_duration = results[-1]['end_sec'] if results else 0

    print("\n" + "=" * 60)
    print(f"INFERENCE SUMMARY: {os.path.basename(npy_path)}")
    print("=" * 60)
    print(f"Total duration: {total_duration} seconds")
    print(f"Total windows: {len(results)}")
    print(f"\nAnomaly Score Statistics:")
    print(f"  Min:    {np.min(scores):.4f}")
    print(f"  Max:    {np.max(scores):.4f}")
    print(f"  Mean:   {np.mean(scores):.4f}")
    print(f"  Std:    {np.std(scores):.4f}")

    anomalies = find_anomaly_segments(results, threshold)
    high_count = sum(1 for s in scores if s >= threshold)

    print(f"\nAnomalies (threshold={threshold}):")
    print(f"  Windows above threshold: {high_count} ({100*high_count/len(scores):.1f}%)")
    print(f"  Contiguous anomaly events: {len(anomalies)}")

    if anomalies:
        print(f"\nDetected Anomaly Events:")
        for i, a in enumerate(anomalies):
            print(f"  [{i+1}] {a['start_sec']}s - {a['end_sec']}s "
                  f"(duration: {a['duration']:.0f}s, max_score: {a['max_score']:.3f})")

    print("=" * 60 + "\n")


def infer_single(
    checkpoint_path: str,
    npy_path: str,
    embed_dim: int = 512,
    temporal_agg: str = "mean",
    dropout_rate: float = 0.5,
    model_type: str = "linear",
    unit_duration: int = 2,
    overlap_ratio: float = 0.0,
    batch_size: int = 64,
    output_dir: str = "./inference_etri",
    threshold: float = 0.5,
    visualize: bool = True,
):
    """Run inference on a single npy file."""
    device = get_device()

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

    print(f"Loading checkpoint from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model=model, device=device)
    model = model.to(device)

    print(f"\nProcessing: {npy_path}")
    results = process_npy(
        model, npy_path, device,
        unit_duration=unit_duration,
        overlap_ratio=overlap_ratio,
        batch_size=batch_size,
        model_type=model_type,
    )

    if not results:
        print("No results generated")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(npy_path))[0]

    csv_path = os.path.join(output_dir, f"{file_stem}_scores.csv")
    save_results_csv(results, csv_path, file_stem)
    print_summary(results, npy_path, threshold)

    if visualize:
        viz_path = os.path.join(output_dir, f"{file_stem}_timeline.png")
        visualize_results(results, file_stem, viz_path, threshold)

    return results


def batch_process(
    model: torch.nn.Module,
    npy_paths: List[str],
    device: torch.device,
    output_dir: str,
    unit_duration: int = 2,
    overlap_ratio: float = 0.0,
    batch_size: int = 64,
    threshold: float = 0.5,
    visualize: bool = True,
    model_type: str = "linear",
):
    """Process multiple npy files."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for npy_path in tqdm(npy_paths, desc="Processing files"):
        file_stem = os.path.splitext(os.path.basename(npy_path))[0]
        print(f"\nProcessing: {file_stem}")

        results = process_npy(
            model, npy_path, device,
            unit_duration=unit_duration,
            overlap_ratio=overlap_ratio,
            batch_size=batch_size,
            model_type=model_type,
        )

        if not results:
            continue

        csv_path = os.path.join(output_dir, f"{file_stem}_scores.csv")
        save_results_csv(results, csv_path, file_stem)
        print_summary(results, npy_path, threshold)

        if visualize:
            viz_path = os.path.join(output_dir, f"{file_stem}_timeline.png")
            visualize_results(results, file_stem, viz_path, threshold)

        for r in results:
            r['file_name'] = file_stem
        all_results.extend(results)

    if all_results:
        combined_path = os.path.join(output_dir, "all_results.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with FeatureSpottingModel on npy features (ETRI)"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Model
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

    # Input
    parser.add_argument("--npy", type=str, default=None,
                        help="Path to a single .npy feature file")
    parser.add_argument("--npy-dir", type=str, default=None,
                        help="Path to directory of .npy feature files (recursive)")

    # Window
    parser.add_argument("--unit-duration", type=int, default=2,
                        help="Window size in seconds")
    parser.add_argument("--overlap-ratio", type=float, default=0.0,
                        help="Sliding window overlap ratio (0.0 = no overlap)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./inference_etri",
                        help="Directory to save outputs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Anomaly threshold for segment detection")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Feature-based Anomaly Action Spotting - Inference")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Temporal agg: {args.temporal_agg}")
    print(f"Unit duration: {args.unit_duration}s")
    print(f"Overlap ratio: {args.overlap_ratio}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    if args.npy:
        infer_single(
            checkpoint_path=args.checkpoint,
            npy_path=args.npy,
            embed_dim=args.embed_dim,
            temporal_agg=args.temporal_agg,
            dropout_rate=args.dropout,
            model_type=args.model_type,
            unit_duration=args.unit_duration,
            overlap_ratio=args.overlap_ratio,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            threshold=args.threshold,
            visualize=not args.no_visualize,
        )

    elif args.npy_dir:
        device = get_device()

        if args.model_type == "resmlp":
            model = ResidualMLPSpottingModel(
                embed_dim=args.embed_dim,
                dropout_rate=args.dropout,
                temporal_agg=args.temporal_agg,
            )
        else:
            model = FeatureSpottingModel(
                embed_dim=args.embed_dim,
                num_classes=2,
                dropout_rate=args.dropout,
                temporal_agg=args.temporal_agg,
            )

        print(f"Loading checkpoint from {args.checkpoint}...")
        load_checkpoint(args.checkpoint, model=model, device=device)
        model = model.to(device)

        # Collect all npy files recursively
        npy_paths = []
        for root, dirs, files in os.walk(args.npy_dir):
            for f in sorted(files):
                if f.endswith('.npy'):
                    npy_paths.append(os.path.join(root, f))

        if not npy_paths:
            print(f"No .npy files found in {args.npy_dir}")
            return

        print(f"Found {len(npy_paths)} .npy files")

        batch_process(
            model=model,
            npy_paths=npy_paths,
            device=device,
            output_dir=args.output_dir,
            unit_duration=args.unit_duration,
            overlap_ratio=args.overlap_ratio,
            batch_size=args.batch_size,
            threshold=args.threshold,
            visualize=not args.no_visualize,
            model_type=args.model_type,
        )

    else:
        print("Error: Please specify either --npy or --npy-dir")


if __name__ == "__main__":
    main()
