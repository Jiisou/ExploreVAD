#!/usr/bin/env python3
"""
Real-time Inference Script for Anomaly Action Spotting Model.

Features:
- Process untrimmed videos with non-overlapping 1-second windows
- Output anomaly scores per segment
- Support for single video or batch processing
- Export results to CSV (timestamp, anomaly_score)
- Optional visualization of anomaly scores over time
"""

import argparse
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    SpottingModelConfig,
    SpottingDataConfig,
    get_spotting_config,
    FRAMES_PER_UNIT,
)
from dataset import UCFCrimeSpottingInferenceDataset
from model import create_spotting_model
from utils import get_device, create_spotting_transform


@torch.no_grad()
def process_video(
    model: torch.nn.Module,
    video_path: str,
    transform,
    device: torch.device,
    batch_size: int = 16,
    fallback_fps: int = 30,
    overlap_ratio: float = 0.0,
    unit_duration: float = 2.0,
) -> List[Dict]:
    """
    Process a single video and return anomaly scores per segment.

    Args:
        model: Trained spotting model.
        video_path: Path to video file.
        transform: Video transform pipeline.
        device: Device to run inference on.
        batch_size: Batch size for processing.
        fallback_fps: Fallback frame rate if video FPS detection fails.
        overlap_ratio: Overlap ratio between windows (0.0 = no overlap).
        unit_duration: Duration of each unit in seconds.

    Returns:
        List of dictionaries with segment information and anomaly scores.
    """
    # Create dataset for this video
    dataset = UCFCrimeSpottingInferenceDataset(
        video_path=video_path,
        transform=transform,
        overlap_ratio=overlap_ratio,
        fallback_fps=fallback_fps,
        unit_duration=unit_duration,
    )

    if len(dataset) == 0:
        print(f"Warning: No segments extracted from {video_path}")
        return []

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for single video
    )

    model.eval()
    results = []

    for inputs, metadata_batch in tqdm(data_loader, desc="Processing", leave=False):
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Get anomaly probabilities
        probs = torch.softmax(outputs, dim=1)
        anomaly_probs = probs[:, 1].cpu().numpy()

        # Unpack batch metadata
        for i in range(len(anomaly_probs)):
            results.append({
                'start_frame': metadata_batch['start_frame'][i].item(),
                'end_frame': metadata_batch['end_frame'][i].item(),
                'start_time': metadata_batch['start_time'][i].item(),
                'end_time': metadata_batch['end_time'][i].item(),
                'anomaly_score': float(anomaly_probs[i]),
            })

    return results


def save_results_csv(
    results: List[Dict],
    output_path: str,
    video_name: Optional[str] = None,
):
    """
    Save inference results to CSV.

    Args:
        results: List of segment results.
        output_path: Path to save CSV file.
        video_name: Optional video name to include.
    """
    df = pd.DataFrame(results)
    if video_name:
        df.insert(0, 'video_name', video_name)

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def visualize_results(
    results: List[Dict],
    video_name: str,
    output_path: str,
    threshold: float = 0.5,
):
    """
    Visualize anomaly scores over time.

    Args:
        results: List of segment results.
        video_name: Name of the video.
        output_path: Path to save visualization.
        threshold: Anomaly threshold for visualization.
    """
    timestamps = [r['start_time'] for r in results]
    scores = [r['anomaly_score'] for r in results]

    plt.figure(figsize=(14, 5))

    # Plot anomaly scores
    plt.subplot(1, 1, 1)
    plt.plot(timestamps, scores, 'b-', linewidth=1.5, alpha=0.8)
    plt.fill_between(timestamps, 0, scores, alpha=0.3)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5,
                label=f'Threshold ({threshold})')

    # Highlight high-score regions
    high_score_mask = np.array(scores) >= threshold
    if np.any(high_score_mask):
        for i, (is_high, t, s) in enumerate(zip(high_score_mask, timestamps, scores)):
            if is_high:
                plt.axvspan(t, t + 1.0, alpha=0.2, color='red')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title(f'Anomaly Detection: {video_name}', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(-0.05, 1.05)
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
    """
    Find contiguous anomaly segments above threshold.

    Args:
        results: List of segment results.
        threshold: Anomaly threshold.
        min_duration: Minimum duration for anomaly segment.

    Returns:
        List of anomaly segment dictionaries.
    """
    anomalies = []
    current_anomaly = None

    for r in results:
        is_anomaly = r['anomaly_score'] >= threshold

        if is_anomaly:
            if current_anomaly is None:
                current_anomaly = {
                    'start_time': r['start_time'],
                    'end_time': r['end_time'],
                    'max_score': r['anomaly_score'],
                    'scores': [r['anomaly_score']],
                }
            else:
                current_anomaly['end_time'] = r['end_time']
                current_anomaly['max_score'] = max(
                    current_anomaly['max_score'],
                    r['anomaly_score']
                )
                current_anomaly['scores'].append(r['anomaly_score'])
        else:
            if current_anomaly is not None:
                # Check duration
                duration = current_anomaly['end_time'] - current_anomaly['start_time']
                if duration >= min_duration:
                    current_anomaly['mean_score'] = np.mean(current_anomaly['scores'])
                    current_anomaly['duration'] = duration
                    del current_anomaly['scores']
                    anomalies.append(current_anomaly)
                current_anomaly = None

    # Handle last segment
    if current_anomaly is not None:
        duration = current_anomaly['end_time'] - current_anomaly['start_time']
        if duration >= min_duration:
            current_anomaly['mean_score'] = np.mean(current_anomaly['scores'])
            current_anomaly['duration'] = duration
            del current_anomaly['scores']
            anomalies.append(current_anomaly)

    return anomalies


def print_summary(
    results: List[Dict],
    video_path: str,
    threshold: float = 0.5,
):
    """Print summary of inference results."""
    scores = [r['anomaly_score'] for r in results]
    total_duration = results[-1]['end_time'] if results else 0

    print("\n" + "=" * 60)
    print(f"INFERENCE SUMMARY: {os.path.basename(video_path)}")
    print("=" * 60)
    print(f"Total duration: {total_duration:.1f} seconds")
    print(f"Total segments: {len(results)}")
    print(f"\nAnomaly Score Statistics:")
    print(f"  Min:    {np.min(scores):.4f}")
    print(f"  Max:    {np.max(scores):.4f}")
    print(f"  Mean:   {np.mean(scores):.4f}")
    print(f"  Std:    {np.std(scores):.4f}")

    # Find anomalies
    anomalies = find_anomaly_segments(results, threshold)
    high_score_count = sum(1 for s in scores if s >= threshold)

    print(f"\nAnomalies (threshold={threshold}):")
    print(f"  Segments above threshold: {high_score_count} ({100*high_score_count/len(scores):.1f}%)")
    print(f"  Contiguous anomaly events: {len(anomalies)}")

    if anomalies:
        print(f"\nDetected Anomaly Events:")
        for i, a in enumerate(anomalies):
            print(f"  [{i+1}] {a['start_time']:.1f}s - {a['end_time']:.1f}s "
                  f"(duration: {a['duration']:.1f}s, max_score: {a['max_score']:.3f})")

    print("=" * 60 + "\n")


def batch_process(
    model: torch.nn.Module,
    video_paths: List[str],
    transform,
    device: torch.device,
    output_dir: str,
    batch_size: int = 16,
    fallback_fps: int = 30,
    overlap_ratio: float = 0.0,
    unit_duration: float = 2.0,
    threshold: float = 0.5,
    visualize: bool = True,
):
    """
    Process multiple videos.

    Args:
        model: Trained spotting model.
        video_paths: List of video paths.
        transform: Video transform pipeline.
        device: Device to run inference on.
        output_dir: Directory to save outputs.
        batch_size: Batch size for processing.
        fallback_fps: Fallback frame rate if video FPS detection fails.
        overlap_ratio: Overlap ratio between windows (0.0 = no overlap).
        unit_duration: Duration of each unit in seconds.
        threshold: Anomaly threshold.
        visualize: Whether to create visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for video_path in tqdm(video_paths, desc="Processing videos"):
        video_name = os.path.basename(video_path)
        print(f"\nProcessing: {video_name}")

        # Process video
        results = process_video(
            model, video_path, transform, device, batch_size,
            fallback_fps=fallback_fps,
            overlap_ratio=overlap_ratio,
            unit_duration=unit_duration,
        )

        if not results:
            continue

        # Save individual CSV
        csv_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_scores.csv")
        save_results_csv(results, csv_path, video_name)

        # Print summary
        print_summary(results, video_path, threshold)

        # Visualize
        if visualize:
            viz_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_timeline.png")
            visualize_results(results, video_name, viz_path, threshold)

        # Collect for batch summary
        for r in results:
            r['video_name'] = video_name
        all_results.extend(results)

    # Save combined results
    if all_results:
        combined_path = os.path.join(output_dir, "all_results.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")


def infer_single(
    checkpoint_path: str,
    video_path: str,
    output_dir: str = "./inference_output",
    threshold: float = 0.5,
    visualize: bool = True,
    fallback_fps: int = 30,
    overlap_ratio: float = 0.0,
    unit_duration: float = 2.0,
):
    """
    Run inference on a single video.

    Args:
        checkpoint_path: Path to model checkpoint.
        video_path: Path to video file.
        output_dir: Directory to save outputs.
        threshold: Anomaly threshold.
        visualize: Whether to create visualization.
        fallback_fps: Fallback frame rate if video FPS detection fails.
        overlap_ratio: Overlap ratio between windows (0.0 = no overlap).
        unit_duration: Duration of each unit in seconds.
    """
    device = get_device()

    # Load model
    model_cfg = get_spotting_config()
    model = create_spotting_model(
        num_classes=model_cfg.num_classes,
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Create transform
    data_cfg = SpottingDataConfig()
    transform = create_spotting_transform(model_cfg, data_cfg, is_train=False)

    # Process video
    print(f"\nProcessing: {video_path}")
    results = process_video(
        model, video_path, transform, device,
        fallback_fps=fallback_fps,
        overlap_ratio=overlap_ratio,
        unit_duration=unit_duration,
    )

    if not results:
        print("No results generated")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)

    # Save CSV
    csv_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_scores.csv")
    save_results_csv(results, csv_path, video_name)

    # Print summary
    print_summary(results, video_path, threshold)

    # Visualize
    if visualize:
        viz_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_timeline.png")
        visualize_results(results, video_name, viz_path, threshold)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with Anomaly Action Spotting Model"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to single video file")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Path to directory of videos")
    parser.add_argument("--output-dir", type=str, default="./inference_output",
                        help="Directory to save outputs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Anomaly threshold")
    parser.add_argument("--fallback-fps", type=int, default=30,
                        help="Fallback frame rate if video FPS detection fails")
    parser.add_argument("--overlap-ratio", type=float, default=0.0,
                        help="Overlap ratio between windows (0.0 = no overlap)")
    parser.add_argument("--unit-duration", type=float, default=2.0,
                        help="Duration of each unit in seconds")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.video:
        # Single video mode
        infer_single(
            checkpoint_path=args.checkpoint,
            video_path=args.video,
            output_dir=args.output_dir,
            threshold=args.threshold,
            visualize=not args.no_visualize,
            fallback_fps=args.fallback_fps,
            overlap_ratio=args.overlap_ratio,
            unit_duration=args.unit_duration,
        )

    elif args.video_dir:
        # Batch mode
        device = get_device()

        # Load model
        model_cfg = get_spotting_config()
        model = create_spotting_model(
            num_classes=model_cfg.num_classes,
            pretrained=False,
            checkpoint_path=args.checkpoint,
            device=device,
        )

        # Create transform
        data_cfg = SpottingDataConfig()
        transform = create_spotting_transform(model_cfg, data_cfg, is_train=False)

        # Find videos
        video_paths = []
        for file in os.listdir(args.video_dir):
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                video_paths.append(os.path.join(args.video_dir, file))

        if not video_paths:
            print(f"No videos found in {args.video_dir}")
            return

        print(f"Found {len(video_paths)} videos")

        # Process
        batch_process(
            model=model,
            video_paths=video_paths,
            transform=transform,
            device=device,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            fallback_fps=args.fallback_fps,
            overlap_ratio=args.overlap_ratio,
            unit_duration=args.unit_duration,
            threshold=args.threshold,
            visualize=not args.no_visualize,
        )

    else:
        print("Error: Please specify either --video or --video-dir")


if __name__ == "__main__":
    main()
