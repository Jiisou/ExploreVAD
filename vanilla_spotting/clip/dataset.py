"""
UCF-CRIME Spotting Dataset for CLIP-based sliding window anomaly detection.

Key differences from X3D variant:
- 10 frames per 2-second unit (not 16)
- Output tensor shape: (C, T, H, W) where T=10
- CLIP normalization applied in transform (not Kinetics)

Strict normal sampling (noise handling):
- In abnormal-class videos, untagged (label=0) windows are only kept if they
  end BEFORE the earliest annotated event. Post-event untagged windows are
  discarded as noisy (ambiguous transition / residual anomaly).
- Multiple annotation rows per video are merged into non-overlapping intervals.
"""

import os
import re
import random
from collections import defaultdict
from typing import Optional, Callable, List, Dict, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

from config import (
    SpottingDataConfig,
    FRAMES_PER_UNIT,
    get_frame_indices,
)


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


class UCFCrimeSpottingDataset(Dataset):
    """
    UCF-CRIME Spotting Dataset for CLIP-based sliding window anomaly detection.

    Extracts 2-second video units from untrimmed clips using a sliding window.
    Each unit yields 10 uniformly sampled frames processed by CLIP.

    Args:
        video_dir: Path to video directory containing class subdirectories.
        annotation_csv: Path to CSV file or directory with annotations.
        transform: Optional transform to apply to video tensors.
        overlap_ratio: Overlap ratio between consecutive windows (0.0 to 1.0).
        fallback_fps: Fallback frame rate if video FPS detection fails.
        unit_duration: Duration of each unit in seconds (default: 2.0).
        max_videos_per_class: Maximum videos per class (None = no limit).
        strict_normal_sampling: If True, in abnormal-class videos only keep
            normal (label=0) windows that end before the earliest event.
            Post-event untagged windows are discarded as noise.
        verbose: Whether to print loading progress.
        seed: Random seed for reproducible video sampling.
    """

    def __init__(
        self,
        video_dir: str,
        annotation_csv: str,
        transform: Optional[Callable] = None,
        overlap_ratio: float = 0.5,
        fallback_fps: int = 30,
        unit_duration: float = 2.0,
        max_videos_per_class: Optional[int] = None,
        strict_normal_sampling: bool = True,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.video_dir = video_dir
        self.annotation_csv = annotation_csv
        self.transform = transform
        self.overlap_ratio = overlap_ratio
        self.fallback_fps = fallback_fps
        self.unit_duration = unit_duration
        self.max_videos_per_class = max_videos_per_class
        self.strict_normal_sampling = strict_normal_sampling
        self.verbose = verbose
        self.seed = seed

        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}")

        if not DECORD_AVAILABLE and not AV_AVAILABLE:
            raise ImportError("Please install decord or av: pip install decord or pip install av")

        self._fps_stats: List[float] = []
        self._discarded_post_event: int = 0  # noisy post-event normal windows discarded
        self.annotations: Dict[str, List[Tuple[float, float]]] = {}
        self.samples: List[Dict] = []

        self._load_annotations()
        self._build_samples()

        if self.verbose:
            self._print_statistics()

    def _load_annotations(self):
        """Load annotations from CSV files."""
        if os.path.isdir(self.annotation_csv):
            self._load_annotations_from_dir(self.annotation_csv)
        else:
            self._load_annotations_from_file(self.annotation_csv)

    def _load_annotations_from_dir(self, annotation_dir: str):
        """Load annotations from directory of CSV files."""
        for csv_file in os.listdir(annotation_dir):
            if not csv_file.endswith('.csv'):
                continue
            csv_path = os.path.join(annotation_dir, csv_file)
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip')
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read {csv_path}: {e}")
                continue
            self._parse_annotation_df(df)

    def _load_annotations_from_file(self, csv_path: str):
        """Load annotations from single CSV file."""
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            self._parse_annotation_df(df)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not read {csv_path}: {e}")

    def _parse_annotation_df(self, df: pd.DataFrame):
        """Parse annotation DataFrame."""
        for _, row in df.iterrows():
            file_name = str(row['file_name']).strip().split()[0]
            start_time = time_to_seconds(row.get('start_time', 0))
            end_time = time_to_seconds(row.get('end_time', 0))

            if file_name not in self.annotations:
                self.annotations[file_name] = []

            if end_time > start_time:
                self.annotations[file_name].append((start_time, end_time))

    def _build_samples(self):
        """Build sample index by scanning videos and creating sliding windows."""
        videos_by_class: Dict[str, List[str]] = defaultdict(list)
        for root, dirs, files in os.walk(self.video_dir):
            if '00_timestamp' in root:
                continue
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    video_path = os.path.join(root, file)
                    class_name = os.path.basename(os.path.dirname(video_path))
                    videos_by_class[class_name].append(video_path)

        # Apply video balancing
        video_files = []
        if self.max_videos_per_class is not None:
            rng = random.Random(self.seed)
            if self.verbose:
                print(f"\nBalancing videos: max {self.max_videos_per_class} per class")
            for class_name, class_videos in sorted(videos_by_class.items()):
                n_original = len(class_videos)
                if n_original > self.max_videos_per_class:
                    sampled = rng.sample(class_videos, self.max_videos_per_class)
                else:
                    sampled = class_videos
                video_files.extend(sampled)
                if self.verbose:
                    print(f"  {class_name}: {len(sampled)}/{n_original} videos")
        else:
            for class_videos in videos_by_class.values():
                video_files.extend(class_videos)

        if self.verbose:
            print(f"Found {len(video_files)} video files (from {len(videos_by_class)} classes)")

        video_iter = tqdm(video_files, desc="Processing videos") if self.verbose else video_files
        for video_path in video_iter:
            self._process_video(video_path)

    @staticmethod
    def _merge_events(
        events: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Merge overlapping or adjacent event intervals.

        Handles the case where a single video has multiple annotation rows
        (e.g. two separate anomaly events, or one event split across rows).
        Overlapping/touching intervals are merged into one.

        Example:
            [(5, 20), (15, 30), (50, 60)] -> [(5, 30), (50, 60)]
        """
        if len(events) <= 1:
            return events

        sorted_events = sorted(events, key=lambda x: x[0])
        merged = [sorted_events[0]]

        for start, end in sorted_events[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:  # overlapping or adjacent
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return merged

    def _process_video(self, video_path: str):
        """
        Process a single video and add sliding window samples.

        Strict normal sampling (when enabled):
            In abnormal-class videos, untagged windows are only adopted if
            they end before the earliest annotated event start. This removes
            noisy post-event segments where the scene may still contain
            residual anomaly cues despite not being annotated.

            Timeline illustration (abnormal video):
            |--normal (keep)--|==EVENT 1==|--discard--|==EVENT 2==|--discard--|
                              ^ earliest_event_start
        """
        video_name = os.path.basename(video_path)

        # Collect all annotations matching this video (handles multiple rows)
        events = []
        for key in self.annotations:
            if key in video_path or key == video_name:
                events.extend(self.annotations[key])

        if not events:
            return

        # Merge overlapping/adjacent events from multiple annotation rows
        events = self._merge_events(events)

        # Get video info
        try:
            if DECORD_AVAILABLE:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                video_fps = vr.get_avg_fps()
            else:
                container = av.open(video_path)
                stream = container.streams.video[0]
                total_frames = stream.frames
                video_fps = float(stream.average_rate)
                container.close()

            if video_fps <= 0:
                video_fps = self.fallback_fps

            self._fps_stats.append(video_fps)

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not read video {video_path}: {e}")
            return

        parent_dir = os.path.basename(os.path.dirname(video_path))
        is_normal_class = parent_dir.lower() == 'normal'

        # Earliest event boundary (for strict normal sampling)
        earliest_event_start = min(e[0] for e in events)

        # Sliding window
        window_size = int(video_fps * self.unit_duration)
        stride = max(1, int(window_size * (1.0 - self.overlap_ratio)))
        num_windows = max(0, (total_frames - window_size) // stride + 1)

        for i in range(num_windows):
            start_frame = i * stride
            end_frame = start_frame + window_size

            start_time = start_frame / video_fps
            end_time = end_frame / video_fps

            # Label: 1 if window overlaps any merged anomaly event
            label = 0
            for event_start, event_end in events:
                if start_time < event_end and end_time > event_start:
                    label = 1
                    break

            # Strict normal sampling: in abnormal videos, discard untagged
            # windows that occur at or after the first event start.
            # Only pre-event normal windows are clean enough to use.
            if (self.strict_normal_sampling
                    and not is_normal_class
                    and label == 0
                    and end_time > earliest_event_start):
                self._discarded_post_event += 1
                continue

            self.samples.append({
                'video_path': video_path,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'label': label,
                'video_fps': video_fps,
            })

    def _print_statistics(self):
        """Print dataset statistics."""
        total = len(self.samples)
        if total == 0:
            print("\nNo samples found in dataset.\n")
            return

        normal = sum(1 for s in self.samples if s['label'] == 0)
        abnormal = total - normal

        if self._fps_stats:
            min_fps = min(self._fps_stats)
            max_fps = max(self._fps_stats)
            avg_fps = sum(self._fps_stats) / len(self._fps_stats)
            fps_info = f"{avg_fps:.1f} avg (range: {min_fps:.1f}-{max_fps:.1f})"
        else:
            fps_info = "N/A"

        print(f"\n{'='*50}")
        print(f"CLIP Spotting Dataset")
        print(f"{'='*50}")
        print(f"Total samples: {total}")
        print(f"  Normal:   {normal} ({100*normal/total:.1f}%)")
        print(f"  Abnormal: {abnormal} ({100*abnormal/total:.1f}%)")
        if self.strict_normal_sampling and self._discarded_post_event > 0:
            print(f"  Discarded (post-event noise): {self._discarded_post_event}")
        print(f"Unit duration: {self.unit_duration}s")
        print(f"Frames per unit: {FRAMES_PER_UNIT}")
        print(f"Overlap ratio: {self.overlap_ratio:.1%}")
        print(f"Strict normal sampling: {self.strict_normal_sampling}")
        print(f"Video FPS: {fps_info}")
        print(f"{'='*50}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (video_tensor [C, T, H, W], label).
        """
        max_retries = 10
        original_idx = idx

        for retry in range(max_retries):
            try:
                sample = self.samples[idx]

                frames = self._extract_frames(
                    sample['video_path'],
                    sample['start_frame'],
                    sample['end_frame'],
                )

                # Stack: (T, H, W, C)
                video_tensor = torch.from_numpy(np.stack(frames)).float()

                if self.transform is not None:
                    video_tensor = self.transform(video_tensor)

                return video_tensor, sample['label']

            except Exception as e:
                if retry == 0:
                    print(f"Warning: Error loading sample {idx} "
                          f"({sample['video_path']}): {type(e).__name__}")
                idx = np.random.randint(0, len(self.samples))

        raise RuntimeError(f"Failed to load sample after {max_retries} retries. "
                          f"Original idx: {original_idx}")

    def _extract_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
    ) -> List[np.ndarray]:
        """
        Extract and sample frames from video.

        Returns:
            List of FRAMES_PER_UNIT (10) sampled frames as numpy arrays.
        """
        window_size = end_frame - start_frame
        frame_indices = get_frame_indices(window_size, FRAMES_PER_UNIT)

        absolute_indices = [start_frame + i for i in frame_indices]

        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            max_idx = len(vr) - 1
            absolute_indices = [min(max(0, i), max_idx) for i in absolute_indices]
            frames = vr.get_batch(absolute_indices).asnumpy()
            frames = list(frames)
        else:
            container = av.open(video_path)
            stream = container.streams.video[0]

            total_frames = stream.frames
            if total_frames <= 0:
                total_frames = int(stream.duration * stream.average_rate / stream.time_base)

            max_idx = max(0, total_frames - 1)
            absolute_indices = [min(max(0, i), max_idx) for i in absolute_indices]

            all_frames = []
            max_needed = max(absolute_indices) + 1
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_needed:
                    break
                all_frames.append(frame.to_ndarray(format='rgb24'))
            container.close()

            if len(all_frames) == 0:
                raise ValueError(f"No frames decoded from {video_path}")

            max_idx = len(all_frames) - 1
            absolute_indices = [min(i, max_idx) for i in absolute_indices]
            frames = [all_frames[i] for i in absolute_indices]

        return frames

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading video."""
        return self.samples[idx].copy()

    def get_video_samples(self, video_path: str) -> List[int]:
        """Get all sample indices for a specific video."""
        return [i for i, s in enumerate(self.samples) if s['video_path'] == video_path]

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        labels = [s['label'] for s in self.samples]
        from utils import compute_class_weights
        return compute_class_weights(labels, num_classes=2)


class NPYFeatureDataset(Dataset):
    """
    Pre-extracted Feature Dataset for training on pre-extracted npy features.

    Each .npy file has shape (T_seconds, embed_dim) where row i is the
    feature vector at second i. Sliding windows of `unit_duration` seconds
    are created and labeled based on annotation overlap.

    Strict normal sampling: since all classes are anomaly types, normal
    (untagged) windows are only kept if they end before the earliest event.

    Args:
        feature_dir: Path to feature directory containing class subdirectories.
        annotation_dir: Path to directory with {class}_timestamp.csv files.
        unit_duration: Duration of each window in seconds (default: 2).
        overlap_ratio: Overlap ratio between consecutive windows (0.0 to 1.0).
        strict_normal_sampling: If True, discard post-event normal windows.
        max_files_per_class: Maximum files per class (None = no limit).
        verbose: Whether to print loading progress.
        seed: Random seed for reproducible file sampling.
    """

    def __init__(
        self,
        feature_dir: str,
        annotation_path: str,
        # txt_file_path="/mnt/c/Users/USER/Desktop/ExploreVAD/annotation/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt",
        unit_duration: int = 2,
        overlap_ratio: float = 0.5,
        strict_normal_sampling: bool = True,
        max_files_per_class: Optional[int] = None,
        fps: float=30.0,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.feature_dir = feature_dir
        self.annotation_path = annotation_path
        self.unit_duration = unit_duration
        self.overlap_ratio = overlap_ratio
        self.strict_normal_sampling = strict_normal_sampling
        self.max_files_per_class = max_files_per_class
        self.verbose = verbose
        self.seed = seed
        self.annotation_path = annotation_path
        self.fps = float(fps)
    

        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}")

        self._discarded_post_event: int = 0
        self.annotations: Dict[str, List[Tuple[float, float]]] = {}
        self.samples: List[Dict] = []

        # self._load_annotations()
        self._load_annotation_txt()
        self._build_samples()

        if self.verbose:
            self._print_statistics()

    def _load_annotations(self):
        """Load annotations from all CSV files in annotation_dir."""
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
                # Remove video extension if present (.mp4, .avi, etc.) for matching with npy stems
                file_stem = os.path.splitext(file_name)[0]
                start_time = time_to_seconds(row.get('start_time', 0))
                end_time = time_to_seconds(row.get('end_time', 0))

                if end_time > start_time:
                    if file_stem not in self.annotations:
                        self.annotations[file_stem] = []
                    self.annotations[file_stem].append((start_time, end_time))

        if self.verbose:
            print(f"Loaded annotations for {len(self.annotations)} files")

    def _load_annotation_txt(self):
        """
        TXT line example:
        Abuse028_x264.mp4 Abuse 165 240 -1 -1

        columns:
        0 file_name
        1 class_name (unused for labeling, but can be stored)
        2 start_frame_1
        3 end_frame_1
        4 start_frame_2 (or -1)
        5 end_frame_2 (or -1)
        """
        self.annotations = {}  # file_stem -> list[(start_sec, end_sec)]
        self.video_class = {}  # optional: file_stem -> class string

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    if self.verbose:
                        print(f"[WARN] line {line_no}: too few columns -> {line}")
                    continue

                file_name = parts[0].strip()
                cls = parts[1].strip() if len(parts) >= 2 else "unknown"

                # Parse frames (robust)
                def _to_int(x, default=-1):
                    try:
                        return int(x)
                    except:
                        return default

                s1 = _to_int(parts[2])
                e1 = _to_int(parts[3])
                s2 = _to_int(parts[4]) if len(parts) >= 6 else -1
                e2 = _to_int(parts[5]) if len(parts) >= 6 else -1

                file_stem = os.path.splitext(os.path.basename(file_name))[0]
                self.video_class[file_stem] = cls

                events = []
                if s1 >= 0 and e1 >= 0 and e1 >= s1:
                    events.append((s1, e1))
                if s2 >= 0 and e2 >= 0 and e2 >= s2:
                    events.append((s2, e2))

                if not events:
                    # no valid events -> you may want to treat as normal video,
                    # but in your design, normal videos usually live in Normal/ folder.
                    continue

                # Convert to seconds (end inclusive -> +1 frame)
                sec_events = []
                for sf, ef in events:
                    start_sec = sf / self.fps
                    end_sec = (ef + 1) / self.fps
                    if end_sec > start_sec:
                        sec_events.append((start_sec, end_sec))

                if sec_events:
                    self.annotations[file_stem] = sec_events

        if self.verbose:
            print(f"Loaded TXT annotations for {len(self.annotations)} files")


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

        # Apply file balancing
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
        """
        Process a single npy feature file and add sliding window samples.

        Each row in the npy array represents 1 second of video.
        Windows of `unit_duration` consecutive seconds are created.

        Strict normal sampling:
            In anomalous videos, normal (untagged) windows are only kept
            if they end before the earliest annotated event start.

            Timeline:
            |--normal (keep)--|==EVENT==|--discard--|
                              ^ earliest_event_start

        Normal class handling:
            Files in 'normal' or 'Normal' directories (or without annotations)
            are treated as entirely normal (label=0 for all windows).
        """
        file_stem = os.path.splitext(os.path.basename(npy_path))[0]
        parent_dir = os.path.basename(os.path.dirname(npy_path))
        is_normal_class = parent_dir.lower() == 'normal'

        # Look up annotation
        events = self.annotations.get(file_stem, [])

        # Get feature duration from npy shape without loading full array
        feat = np.load(npy_path, mmap_mode='r')
        # total_seconds = feat.shape[0]
        T = feat.shape[0]  # <= "seconds"가 아니라 "rows"


        if T < self.unit_duration:
            return

        # If no events and not normal class, skip (missing annotation)
        if not events and not is_normal_class:
            return

        # Merge overlapping events
        if events:
            events = UCFCrimeSpottingDataset._merge_events(events)
            earliest_event_start = min(e[0] for e in events)
        else:
            earliest_event_start = float('inf')  # No events for normal class

        # Sliding window over seconds
        stride = max(1, int(self.unit_duration * (1.0 - self.overlap_ratio)))  # row stride
        num_windows = max(0, (T - self.unit_duration) // stride + 1)    

        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + self.unit_duration  # slice rows

            # Label: 1 if window overlaps any event, 0 otherwise
            label = 0
            if not is_normal_class:
                for event_start, event_end in events:
                    if start_idx < event_end and end_idx > event_start:
                        label = 1
                        break
            
        def row_to_time_interval(r: int) -> tuple[float, float]:
            # row 정의[r-1, r+1)(0-index)에 맞춰 r<=1 예외 처리
            start = 0.0 if r <= 1 else float(r - 1)
            end = start + 2.0
            return start, end

        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + self.unit_duration

            seg_start_time = row_to_time_interval(start_idx)[0]
            seg_end_time = row_to_time_interval(end_idx - 1)[1]

            label = 0
            if not is_normal_class:
                for event_start, event_end in events:   # seconds
                    if seg_start_time < event_end and seg_end_time > event_start:
                        label = 1
                        break

            if (self.strict_normal_sampling
                    and not is_normal_class
                    and label == 0
                    and seg_end_time > earliest_event_start):
                self._discarded_post_event += 1
                continue

            self.samples.append({
                'npy_path': npy_path,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'label': label,
                'seg_start_time': seg_start_time, # 디버깅 편의
                'seg_end_time': seg_end_time,     
            })

    def _print_statistics(self):
        """Print dataset statistics."""
        total = len(self.samples)
        if total == 0:
            print("\nNo samples found in dataset.\n")
            return

        normal = sum(1 for s in self.samples if s['label'] == 0)
        abnormal = total - normal

        print(f"\n{'='*50}")
        print(f"NPY Feature Dataset (Pre-extracted video features)")
        print(f"{'='*50}")
        print(f"Total samples: {total}")
        print(f"  Normal:   {normal} ({100*normal/total:.1f}%)")
        print(f"  Abnormal: {abnormal} ({100*abnormal/total:.1f}%)")
        if self.strict_normal_sampling and self._discarded_post_event > 0:
            print(f"  Discarded (post-event noise): {self._discarded_post_event}")
        print(f"Unit duration: {self.unit_duration}s")
        print(f"Overlap ratio: {self.overlap_ratio:.1%}")
        print(f"Strict normal sampling: {self.strict_normal_sampling}")
        print(f"{'='*50}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Returns:
            Tuple of (feature_tensor [T_window, D], label).
        """
        sample = self.samples[idx]
        feat = np.load(sample['npy_path'], mmap_mode='r')
        # window = feat[sample['start_sec']:sample['end_sec']]  # (unit_duration, D)
        window = feat[sample['start_idx']:sample['end_idx']]
        feature_tensor = torch.from_numpy(np.array(window)).float()
        return feature_tensor, sample['label']

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample."""
        return self.samples[idx].copy()

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        labels = [s['label'] for s in self.samples]
        from utils import compute_class_weights
        return compute_class_weights(labels, num_classes=2)


class UCFCrimeSpottingInferenceDataset(Dataset):
    """
    Simplified dataset for inference on a single video.

    Args:
        video_path: Path to video file.
        transform: Optional transform to apply.
        overlap_ratio: Overlap ratio between windows (0.0 for inference).
        fallback_fps: Fallback frame rate.
        unit_duration: Duration of each unit in seconds.
    """

    def __init__(
        self,
        video_path: str,
        transform: Optional[Callable] = None,
        overlap_ratio: float = 0.0,
        fallback_fps: int = 30,
        unit_duration: float = 2.0,
    ):
        self.video_path = video_path
        self.transform = transform
        self.overlap_ratio = overlap_ratio
        self.fallback_fps = fallback_fps
        self.unit_duration = unit_duration

        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}")

        # Get video info
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            self.total_frames = len(vr)
            self.video_fps = vr.get_avg_fps()
        else:
            container = av.open(video_path)
            stream = container.streams.video[0]
            self.total_frames = stream.frames
            self.video_fps = float(stream.average_rate)
            container.close()

        if self.video_fps <= 0:
            self.video_fps = fallback_fps

        self.window_size = int(self.video_fps * unit_duration)
        self.stride = max(1, int(self.window_size * (1.0 - overlap_ratio)))
        self.num_windows = max(0, (self.total_frames - self.window_size) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a window.

        Returns:
            Tuple of (video_tensor [C, T, H, W], metadata dict).
        """
        start_frame = idx * self.stride
        end_frame = start_frame + self.window_size

        window_size = end_frame - start_frame
        frame_indices = get_frame_indices(window_size, FRAMES_PER_UNIT)
        absolute_indices = [start_frame + i for i in frame_indices]

        if DECORD_AVAILABLE:
            vr = VideoReader(self.video_path, ctx=cpu(0))
            max_idx = len(vr) - 1
            absolute_indices = [min(i, max_idx) for i in absolute_indices]
            frames = vr.get_batch(absolute_indices).asnumpy()
        else:
            container = av.open(self.video_path)
            all_frames = []
            for frame in container.decode(video=0):
                all_frames.append(frame.to_ndarray(format='rgb24'))
            container.close()
            max_idx = len(all_frames) - 1
            absolute_indices = [min(i, max_idx) for i in absolute_indices]
            frames = np.stack([all_frames[i] for i in absolute_indices])

        video_tensor = torch.from_numpy(frames).float()

        if self.transform is not None:
            video_tensor = self.transform(video_tensor)

        metadata = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_frame / self.video_fps,
            'end_time': end_frame / self.video_fps,
            'video_fps': self.video_fps,
            'window_size': self.window_size,
        }

        return video_tensor, metadata
