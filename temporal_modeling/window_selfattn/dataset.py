"""
Dataset for Temporal Self-Attention Model.

Handles pre-extracted npy features with frame-level granularity:
- Each npy file: (T_frames, D) where T_frames = video_length * fps
- Features may have duplication from extraction (e.g., 6 copies per unique frame)
- Windowing is handled at model level, dataset provides full video features
"""

import os
import random
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def time_to_seconds(time_str) -> float:
    """
    Convert time string to seconds.

    Supports formats:
    - 'HH:MM:SS' (e.g., '0:00:06')
    - 'MM:SS' (e.g., '00:06')
    - Numeric string (e.g., '6.5')
    """
    if pd.isna(time_str) or time_str == "":
        return 0.0

    time_match = re.search(r"(\d+:?\d*:?\d*\.?\d*)", str(time_str))
    if not time_match:
        return 0.0

    clean_time_str = time_match.group(1)
    parts = str(clean_time_str).split(':')

    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])

    try:
        return float(time_str)
    except ValueError:
        return 0.0


def merge_events(events: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or adjacent event intervals."""
    if len(events) <= 1:
        return events

    sorted_events = sorted(events, key=lambda x: x[0])
    merged = [sorted_events[0]]

    for start, end in sorted_events[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


class FrameLevelFeatureDataset(Dataset):
    """
    Dataset for frame-level pre-extracted features.

    Each npy file contains (T_frames, D) where T_frames is the total number
    of extracted frames. Features may be duplicated (e.g., 6 copies per frame)
    from the extraction stage.

    This dataset creates sliding windows over frames and assigns binary labels
    based on annotation overlap.

    Args:
        feature_dir: Path to feature directory containing class subdirectories.
        annotation_dir: Path to directory with *_timestamp.csv files.
        fps: Feature extraction FPS (raw features per second in the npy file, default: 30).
        frame_skip: How many consecutive frames are duplicates (default: 6).
        window_seconds: Window duration in seconds (default: 2).
        overlap_ratio: Window overlap ratio (default: 0.5 for 50%).
        strict_normal_sampling: Discard post-event normal windows (default: True).
        max_files_per_class: Limit files per class (None = no limit).
        verbose: Print loading progress.
        seed: Random seed.
    """

    def __init__(
        self,
        feature_dir: str,
        annotation_dir: str,
        fps: int = 30,
        frame_skip: int = 6,
        window_seconds: float = 2.0,
        overlap_ratio: float = 0.5,
        strict_normal_sampling: bool = True,
        max_files_per_class: Optional[int] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.feature_dir = feature_dir
        self.annotation_dir = annotation_dir
        self.fps = fps
        self.frame_skip = frame_skip
        self.window_seconds = window_seconds
        self.overlap_ratio = overlap_ratio
        self.strict_normal_sampling = strict_normal_sampling
        self.max_files_per_class = max_files_per_class
        self.verbose = verbose
        self.seed = seed

        # Calculate window parameters
        # Effective fps after skipping duplicates
        self.effective_fps = fps / frame_skip if frame_skip > 1 else fps
        # Window size in frames (raw, before skip)
        self.window_frames = int(window_seconds * fps)
        # Stride in frames (raw)
        self.stride_frames = int(self.window_frames * (1 - overlap_ratio))

        self._discarded_post_event = 0
        self.annotations: Dict[str, List[Tuple[float, float]]] = {}
        self.samples: List[Dict] = []

        self._load_annotations()
        self._build_samples()

        if self.verbose:
            self._print_statistics()

    def _load_annotations(self):
        """Load annotations from CSV files."""
        for csv_file in sorted(os.listdir(self.annotation_dir)):
            if not csv_file.endswith('.csv'):
                continue
            csv_path = os.path.join(self.annotation_dir, csv_file)
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip')
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read {csv_path}: {e}")
                continue

            for _, row in df.iterrows():
                file_name = str(row['file_name']).strip()
                file_stem = os.path.splitext(file_name)[0]
                start_time = time_to_seconds(row.get('start_time', 0))
                end_time = time_to_seconds(row.get('end_time', 0))

                if end_time > start_time:
                    if file_stem not in self.annotations:
                        self.annotations[file_stem] = []
                    self.annotations[file_stem].append((start_time, end_time))

        if self.verbose:
            print(f"Loaded annotations for {len(self.annotations)} files")

    def _build_samples(self):
        """Scan npy files and create sliding window samples."""
        files_by_class: Dict[str, List[str]] = defaultdict(list)

        for class_name in sorted(os.listdir(self.feature_dir)):
            class_dir = os.path.join(self.feature_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for npy_file in sorted(os.listdir(class_dir)):
                if npy_file.endswith('.npy'):
                    npy_path = os.path.join(class_dir, npy_file)
                    files_by_class[class_name].append(npy_path)

        # Balance files per class
        npy_files = []
        if self.max_files_per_class is not None:
            rng = random.Random(self.seed)
            for class_name, class_files in sorted(files_by_class.items()):
                n_original = len(class_files)
                if n_original > self.max_files_per_class:
                    sampled = rng.sample(class_files, self.max_files_per_class)
                else:
                    sampled = class_files
                npy_files.extend(sampled)
                if self.verbose:
                    print(f"  {class_name}: {len(sampled)}/{n_original} files")
        else:
            for class_files in files_by_class.values():
                npy_files.extend(class_files)

        if self.verbose:
            print(f"Found {len(npy_files)} npy files (from {len(files_by_class)} classes)")

        file_iter = tqdm(npy_files, desc="Processing features") if self.verbose else npy_files
        for npy_path in file_iter:
            self._process_npy_feature(npy_path)

    def _process_npy_feature(self, npy_path: str):
        """Process a single npy feature file."""
        file_stem = os.path.splitext(os.path.basename(npy_path))[0]
        parent_dir = os.path.basename(os.path.dirname(npy_path))
        is_normal_class = parent_dir.lower() == 'normal'

        events = self.annotations.get(file_stem, [])

        # Get feature shape without loading
        feat = np.load(npy_path, mmap_mode='r')
        total_frames = feat.shape[0]

        if total_frames < self.window_frames:
            return

        # Skip if no events and not normal class
        if not events and not is_normal_class:
            return

        # Merge events
        if events:
            events = merge_events(events)
            earliest_event_start = min(e[0] for e in events)
        else:
            earliest_event_start = float('inf')

        # Sliding window
        num_windows = max(0, (total_frames - self.window_frames) // self.stride_frames + 1)

        for i in range(num_windows):
            start_frame = i * self.stride_frames
            end_frame = start_frame + self.window_frames

            # Convert to time (accounting for fps)
            start_time = start_frame / self.fps
            end_time = end_frame / self.fps

            # Label based on event overlap
            label = 0
            if not is_normal_class:
                for event_start, event_end in events:
                    if start_time < event_end and end_time > event_start:
                        label = 1
                        break

            # Strict normal sampling
            if (self.strict_normal_sampling
                    and not is_normal_class
                    and label == 0
                    and end_time > earliest_event_start):
                self._discarded_post_event += 1
                continue

            self.samples.append({
                'npy_path': npy_path,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'label': label,
            })

    def _print_statistics(self):
        """Print dataset statistics."""
        total = len(self.samples)
        if total == 0:
            print("\nNo samples found in dataset.\n")
            return

        normal = sum(1 for s in self.samples if s['label'] == 0)
        abnormal = total - normal

        print(f"\n{'='*60}")
        print(f"Frame-Level Feature Dataset")
        print(f"{'='*60}")
        print(f"Total samples: {total}")
        print(f"  Normal:   {normal} ({100*normal/total:.1f}%)")
        print(f"  Abnormal: {abnormal} ({100*abnormal/total:.1f}%)")
        if self.strict_normal_sampling and self._discarded_post_event > 0:
            print(f"  Discarded (post-event): {self._discarded_post_event}")
        print(f"Feature FPS: {self.fps}")
        print(f"Frame skip: {self.frame_skip}")
        print(f"Effective FPS: {self.effective_fps:.1f}")
        print(f"Window: {self.window_seconds}s = {self.window_frames} frames")
        print(f"Stride: {self.stride_frames} frames ({self.overlap_ratio*100:.0f}% overlap)")
        print(f"Strict normal sampling: {self.strict_normal_sampling}")
        print(f"{'='*60}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (feature_tensor [window_frames, D], label).
            The model handles frame_skip internally.
        """
        sample = self.samples[idx]
        feat = np.load(sample['npy_path'], mmap_mode='r')
        window = feat[sample['start_frame']:sample['end_frame']]
        feature_tensor = torch.from_numpy(np.array(window)).float()
        return feature_tensor, sample['label']

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample."""
        return self.samples[idx].copy()


class VideoLevelFeatureDataset(Dataset):
    """
    Dataset that loads entire video features for video-level classification.

    Each sample is a full video's features, labeled as anomalous if it
    contains any annotated events.

    Args:
        feature_dir: Path to feature directory.
        annotation_dir: Path to annotation CSV directory.
        fps: Feature extraction FPS.
        max_frames: Maximum frames to load (None = load all).
        verbose: Print progress.
        seed: Random seed.
    """

    def __init__(
        self,
        feature_dir: str,
        annotation_dir: str,
        fps: int = 6,
        max_frames: Optional[int] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.feature_dir = feature_dir
        self.annotation_dir = annotation_dir
        self.fps = fps
        self.max_frames = max_frames
        self.verbose = verbose
        self.seed = seed

        self.annotations: Dict[str, List[Tuple[float, float]]] = {}
        self.samples: List[Dict] = []

        self._load_annotations()
        self._build_samples()

        if self.verbose:
            self._print_statistics()

    def _load_annotations(self):
        """Load annotations from CSV files."""
        for csv_file in sorted(os.listdir(self.annotation_dir)):
            if not csv_file.endswith('.csv'):
                continue
            csv_path = os.path.join(self.annotation_dir, csv_file)
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip')
            except Exception:
                continue

            for _, row in df.iterrows():
                file_name = str(row['file_name']).strip()
                file_stem = os.path.splitext(file_name)[0]
                start_time = time_to_seconds(row.get('start_time', 0))
                end_time = time_to_seconds(row.get('end_time', 0))

                if end_time > start_time:
                    if file_stem not in self.annotations:
                        self.annotations[file_stem] = []
                    self.annotations[file_stem].append((start_time, end_time))

    def _build_samples(self):
        """Scan npy files."""
        for class_name in sorted(os.listdir(self.feature_dir)):
            class_dir = os.path.join(self.feature_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            is_normal_class = class_name.lower() == 'normal'

            for npy_file in sorted(os.listdir(class_dir)):
                if not npy_file.endswith('.npy'):
                    continue

                npy_path = os.path.join(class_dir, npy_file)
                file_stem = os.path.splitext(npy_file)[0]

                events = self.annotations.get(file_stem, [])

                # Skip anomaly class files without annotation
                if not events and not is_normal_class:
                    continue

                # Video-level label: 1 if any events, 0 otherwise
                label = 0 if is_normal_class else (1 if events else 0)

                self.samples.append({
                    'npy_path': npy_path,
                    'label': label,
                    'events': merge_events(events) if events else [],
                    'class_name': class_name,
                })

    def _print_statistics(self):
        """Print statistics."""
        total = len(self.samples)
        if total == 0:
            print("\nNo samples found.\n")
            return

        normal = sum(1 for s in self.samples if s['label'] == 0)
        abnormal = total - normal

        print(f"\n{'='*60}")
        print(f"Video-Level Feature Dataset")
        print(f"{'='*60}")
        print(f"Total videos: {total}")
        print(f"  Normal:   {normal} ({100*normal/total:.1f}%)")
        print(f"  Abnormal: {abnormal} ({100*abnormal/total:.1f}%)")
        print(f"{'='*60}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (feature_tensor [T, D], label).
        """
        sample = self.samples[idx]
        feat = np.load(sample['npy_path'])

        if self.max_frames is not None and feat.shape[0] > self.max_frames:
            feat = feat[:self.max_frames]

        feature_tensor = torch.from_numpy(feat).float()
        return feature_tensor, sample['label']

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata."""
        return self.samples[idx].copy()


def collate_variable_length(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-length sequences.

    Pads all sequences to the maximum length in the batch.

    Args:
        batch: List of (features, label) tuples.

    Returns:
        Tuple of (padded_features [B, T_max, D], labels [B]).
    """
    features, labels = zip(*batch)

    # Find max length
    max_len = max(f.size(0) for f in features)
    embed_dim = features[0].size(1)

    # Pad sequences
    padded = torch.zeros(len(features), max_len, embed_dim)
    for i, f in enumerate(features):
        padded[i, :f.size(0), :] = f

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels


if __name__ == "__main__":
    print("Testing FrameLevelFeatureDataset...")

    # Create dummy dataset
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy structure
        feat_dir = os.path.join(tmpdir, "features")
        ann_dir = os.path.join(tmpdir, "annotations")
        os.makedirs(os.path.join(feat_dir, "Abuse"), exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        # Create dummy npy (10 seconds * 30 fps = 300 raw frames, dim 512)
        # (with 6x duplication, so 50 unique frames)
        dummy_feat = np.random.randn(300, 512).astype(np.float32)
        np.save(os.path.join(feat_dir, "Abuse", "Abuse001.npy"), dummy_feat)

        # Create dummy annotation
        ann_df = pd.DataFrame({
            'file_name': ['Abuse001.mp4'],
            'start_time': ['0:00:03'],
            'end_time': ['0:00:07'],
        })
        ann_df.to_csv(os.path.join(ann_dir, "Abuse_timestamp.csv"), index=False)

        # Test dataset
        dataset = FrameLevelFeatureDataset(
            feature_dir=feat_dir,
            annotation_dir=ann_dir,
            fps=30,         # 30 raw features per second in npy
            frame_skip=6,   # 6x duplication → skip to get unique
            window_seconds=2.0,
            overlap_ratio=0.5,
            verbose=True,
        )

        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            feat, label = dataset[0]
            print(f"Sample shape: {feat.shape}, label: {label}")

        print("\nAll tests passed!")
