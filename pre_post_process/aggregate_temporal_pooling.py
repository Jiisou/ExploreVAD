import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

"""
Usage

* Simple mean aggregation (default)
python aggregate_temporal_snippet.py \
    --input_dir /path/to/frame_features \
    --output_dir /path/to/snippet_features \
    --fps 30

* Top-k mean aggregation (with deduplication)
python aggregate_temporal_snippet.py \
    --input_dir /path/to/frame_features \
    --output_dir /path/to/snippet_features \
    --fps 30 \
    --agg_method topk \
    --top_k 5 \
    --sample_rate 6 \
    --score_method norm

* With per-video FPS from original videos
python aggregate_temporal_snippet.py \
    --input_dir /path/to/frame_features \
    --output_dir /path/to/snippet_features \
    --video_dir /path/to/videos

Aggregation Methods:
  1. mean: Simple mean of all frames in window
  2. topk: Deduplicate copied frames (sample_rate), then top-k mean by score

Deduplication:
  - Feature extraction copies each frame `sample_rate` times (default: 6)
  - Frames [0, 6), [6, 12), [12, 18), ... are duplicates
  - topk method takes only frame 0, 6, 12, ... (unique frames)

Window (window_time=2, stride=1):
  1초 → row 0: aggregation of frames [0s, 1s)
  2초 → row 1: aggregation of frames [0s, 2s)
  3초 → row 2: aggregation of frames [1s, 3s)
  ...
  n초 → row n-1: aggregation of frames [(n-2)s, ns)
"""


def aggregate_frame_features(frame_features, fps, window_time=2):
    """
    Aggregate frame-level features into second-level snippets
    via sliding window mean pooling.

    Args:
        frame_features: [total_frames, feature_dim]
        fps: frames per second
        window_time: window size in seconds (default: 2)

    Returns:
        [duration_seconds, feature_dim]
    """
    total_frames = frame_features.shape[0]
    duration_sec = int(total_frames / fps)

    if duration_sec < 1:
        raise ValueError(
            f"Video too short: {total_frames} frames at {fps} FPS "
            f"= {total_frames / fps:.2f}s"
        )

    snippets = []
    for t in range(1, duration_sec + 1): # stride = 1 고정 (윈도우 2초 기준 overlap 50% 설정된 것과 동일)
        start_frame = int(max(0, t - window_time) * fps)
        end_frame = min(int(t * fps), total_frames)

        window_feats = frame_features[start_frame:end_frame]
        snippets.append(window_feats.mean(axis=0))

    return np.stack(snippets)


def deduplicate_features(frame_features: np.ndarray, sample_rate: int = 6) -> np.ndarray:
    """
    Remove duplicate frames caused by sample_rate copying.

    When extracting features, each sampled frame is copied `sample_rate` times.
    This function takes only unique frames by sampling every `sample_rate`-th frame.

    Args:
        frame_features: [total_frames, feature_dim] with duplicates
        sample_rate: Number of copies per unique frame (default: 6)

    Returns:
        [unique_frames, feature_dim] deduplicated features
    """
    # Take every sample_rate-th frame (first frame of each group)
    return frame_features[::sample_rate]


def aggregate_topk_mean(
    frame_features: np.ndarray,
    fps: float,
    window_time: int = 2,
    top_k: int = 5,
    sample_rate: int = 6,
    score_method: str = "norm",
) -> np.ndarray:
    """
    Aggregate frame-level features into second-level snippets
    via sliding window top-k mean pooling.

    First deduplicates copied frames, then selects top-k features
    within each window based on a scoring method.

    Args:
        frame_features: [total_frames, feature_dim] (with duplicates)
        fps: Original video FPS (before sampling)
        window_time: Window size in seconds (default: 2)
        top_k: Number of top features to average (default: 5)
        sample_rate: Duplicate factor from feature extraction (default: 6)
        score_method: Scoring method for top-k selection
            - "norm": L2 norm (higher = more salient)
            - "var": Variance across dimensions
            - "max": Max value across dimensions

    Returns:
        [duration_seconds, feature_dim]

    Example:
        If original video is 30 FPS and sample_rate=6:
        - Raw features: 30 * T frames, each copied 6 times
        - After dedup: 30/6 * T = 5 unique features per second
        - window_time=2 gives 10 unique features per window
        - top_k=5 selects best 5 and averages them
    """
    # Step 1: Deduplicate copied frames
    unique_features = deduplicate_features(frame_features, sample_rate)

    # Effective FPS after deduplication
    effective_fps = fps / sample_rate
    total_unique_frames = unique_features.shape[0]
    duration_sec = int(total_unique_frames / effective_fps)

    if duration_sec < 1:
        raise ValueError(
            f"Video too short: {total_unique_frames} unique frames at "
            f"{effective_fps:.2f} effective FPS = {total_unique_frames / effective_fps:.2f}s"
        )

    # Step 2: Compute scores for top-k selection
    if score_method == "norm":
        scores = np.linalg.norm(unique_features, axis=1)
    elif score_method == "var":
        scores = np.var(unique_features, axis=1)
    elif score_method == "max":
        scores = np.max(unique_features, axis=1)
    else:
        raise ValueError(f"Unknown score_method: {score_method}")

    # Step 3: Sliding window top-k mean aggregation
    snippets = []
    for t in range(1, duration_sec + 1):
        # Window boundaries in terms of unique frames
        start_frame = int(max(0, t - window_time) * effective_fps)
        end_frame = min(int(t * effective_fps), total_unique_frames)

        window_feats = unique_features[start_frame:end_frame]
        window_scores = scores[start_frame:end_frame]

        if len(window_feats) == 0:
            # Edge case: use zero vector
            snippets.append(np.zeros(unique_features.shape[1]))
            continue

        # Select top-k by score
        k = min(top_k, len(window_feats))
        topk_indices = np.argsort(window_scores)[-k:]  # Top-k indices (highest scores)
        topk_feats = window_feats[topk_indices]

        snippets.append(topk_feats.mean(axis=0))

    return np.stack(snippets)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process frame-level features into second-level "
                    "snippets via sliding window mean pooling"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing frame-level .npy files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save second-level .npy features")
    parser.add_argument("--fps", type=float, default=30,
                        help="Fixed FPS for all videos")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory of original videos to read per-video FPS")
    parser.add_argument("--video-ext", type=str, default=".mp4",
                        help="Video file extension (default: .mp4)")
    parser.add_argument("--window-time", type=int, default=2,
                        help="Window size in seconds (default: 2)")
    parser.add_argument("--agg-method", type=str, default="mean",
                        choices=["mean", "topk"],
                        help="Aggregation method: 'mean' or 'topk' (default: mean)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top features to average for topk method (default: 5)")
    parser.add_argument("--sample-rate", type=int, default=6,
                        help="Duplicate factor from feature extraction (default: 6)")
    parser.add_argument("--score-method", type=str, default="norm",
                        choices=["norm", "var", "max"],
                        help="Scoring method for top-k selection (default: l2-norm)")
    # 이미 feature extraction 단계에서 sampling rate에 맞게 feature 추출됨
    args = parser.parse_args()

    if args.fps is None and args.video_dir is None:
        parser.error("Either --fps or --video_dir must be provided")

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect .npy files
    npy_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".npy"):
                npy_path = os.path.join(root, f)
                rel_path = os.path.relpath(npy_path, args.input_dir)
                npy_files.append(rel_path)

    print(f"Found {len(npy_files)} feature files")

    for rel_path in tqdm(npy_files, desc="Post-processing"):
        input_path = os.path.join(args.input_dir, rel_path)
        output_path = os.path.join(args.output_dir, rel_path)

        frame_features = np.load(input_path)

        # Determine FPS
        if args.fps:
            fps = args.fps
        else:
            video_rel = os.path.splitext(rel_path)[0] + args.video_ext
            video_path = os.path.join(args.video_dir, video_rel)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps <= 0:
                tqdm.write(f"Skipping {rel_path}: could not read FPS from {video_path}")
                continue

        try:
            if args.agg_method == "topk":
                snippets = aggregate_topk_mean(
                    frame_features,
                    fps,
                    window_time=args.window_time,
                    top_k=args.top_k,
                    sample_rate=args.sample_rate,
                    score_method=args.score_method,
                )
            else:
                snippets = aggregate_frame_features(
                    frame_features, fps, args.window_time
                )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, snippets)
            tqdm.write(
                f"Saved {output_path}: {frame_features.shape} -> {snippets.shape}"
            )
        except Exception as e:
            tqdm.write(f"Error processing {rel_path}: {e}")


if __name__ == "__main__":
    main()
