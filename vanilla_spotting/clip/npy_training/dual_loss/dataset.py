"""
UCF-CRIME Spotting Dataset for CLIP-based sliding window anomaly detection.
Modified to return segment-level labels for dual loss training.
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

import sys
# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    SpottingDataConfig,
    FRAMES_PER_UNIT,
    get_frame_indices,
)


def time_to_seconds(time_str) -> float:
    """
    Convert time string to seconds.
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
    UCF-CRIME Spotting Dataset (Original class) - Not modified for dual loss (not used here).
    Kept for compatibility if needed, but mainly focusing on NPYFeatureDataset.
    """
    # ... (omitting strict implementation for this class as NPYFeatureDataset is the target)
    pass


class NPYFeatureDataset(Dataset):
    """
    Pre-extracted Feature Dataset for training on pre-extracted npy features.

    Modified for Dual Loss:
    - Added return_segment_labels=False to __init__
    - If return_segment_labels=True, __getitem__ returns (feature, label, segment_labels)
      segment_labels shape: (T,)
    """

    def __init__(
        self,
        feature_dir: str,
        annotation_dir: str,
        unit_duration: int = 2,
        overlap_ratio: float = 0.5,
        strict_normal_sampling: bool = True,
        max_files_per_class: Optional[int] = None,
        verbose: bool = True,
        seed: int = 42,
        return_segment_labels: bool = False,
    ):
        self.feature_dir = feature_dir
        self.annotation_dir = annotation_dir
        self.unit_duration = unit_duration
        self.overlap_ratio = overlap_ratio
        self.strict_normal_sampling = strict_normal_sampling
        self.max_files_per_class = max_files_per_class
        self.verbose = verbose
        self.seed = seed
        self.return_segment_labels = return_segment_labels

        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}")

        self._discarded_post_event: int = 0
        self.annotations: Dict[str, List[Tuple[float, float]]] = {}
        self.samples: List[Dict] = []

        self._load_annotations()
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
                file_stem = os.path.splitext(file_name)[0]
                start_time = time_to_seconds(row.get('start_time', 0))
                end_time = time_to_seconds(row.get('end_time', 0))

                if end_time > start_time:
                    if file_stem not in self.annotations:
                        self.annotations[file_stem] = []
                    self.annotations[file_stem].append((start_time, end_time))

        if self.verbose:
            print(f"Loaded annotations for {len(self.annotations)} files")

    @staticmethod
    def _merge_events(events):
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
        file_stem = os.path.splitext(os.path.basename(npy_path))[0]
        parent_dir = os.path.basename(os.path.dirname(npy_path))
        is_normal_class = parent_dir.lower() == 'normal'

        events = self.annotations.get(file_stem, [])

        feat = np.load(npy_path, mmap_mode='r')
        total_seconds = feat.shape[0]

        if total_seconds < self.unit_duration:
            return

        if not events and not is_normal_class:
            return

        if events:
            events = self._merge_events(events)
            earliest_event_start = min(e[0] for e in events)
        else:
            earliest_event_start = float('inf')

        stride = max(1, int(self.unit_duration * (1.0 - self.overlap_ratio)))
        num_windows = max(0, (total_seconds - self.unit_duration) // stride + 1)

        for i in range(num_windows):
            start_sec = i * stride
            end_sec = start_sec + self.unit_duration

            label = 0
            seg_labels = [] # Will be populated differently in __getitem__ if needed, but simpler here
            
            # Simple metadata creation. Actual labels computed later or here.
            # To be consistent with existing logic:
            if not is_normal_class:
                for event_start, event_end in events:
                    if start_sec < event_end and end_sec > event_start:
                        label = 1
                        break
            
            # Strict normal sampling
            if (self.strict_normal_sampling
                    and not is_normal_class
                    and label == 0
                    and end_sec > earliest_event_start):
                self._discarded_post_event += 1
                continue

            self.samples.append({
                'npy_path': npy_path,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'label': label,
                'file_stem': file_stem, # Added for quick lookup
                'is_normal_class': is_normal_class
            })

    def _print_statistics(self):
        total = len(self.samples)
        if total == 0:
            print("\nNo samples found in dataset.\n")
            return
        normal = sum(1 for s in self.samples if s['label'] == 0)
        abnormal = total - normal
        print(f"\n{'='*50}")
        print(f"NPY Feature Dataset (Dual Loss)")
        print(f"{'='*50}")
        print(f"Total samples: {total}")
        print(f"  Normal:   {normal} ({100*normal/total:.1f}%)")
        print(f"  Abnormal: {abnormal} ({100*abnormal/total:.1f}%)")
        if self.strict_normal_sampling and self._discarded_post_event > 0:
            print(f"  Discarded (post-event noise): {self._discarded_post_event}")
        print(f"Unit duration: {self.unit_duration}s")
        print(f"Return Segment Labels: {self.return_segment_labels}")
        print(f"{'='*50}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        feat = np.load(sample['npy_path'], mmap_mode='r')
        window = feat[sample['start_sec']:sample['end_sec']]  # (unit_duration, D)
        feature_tensor = torch.from_numpy(np.array(window)).float()
        
        video_label = sample['label']
        
        if self.return_segment_labels:
            start_sec = sample['start_sec']
            end_sec = sample['end_sec']
            file_stem = sample['file_stem']
            is_normal_class = sample['is_normal_class']
            
            # Compute segment labels
            T = end_sec - start_sec
            segment_labels = np.zeros(T, dtype=np.float32)
            
            if not is_normal_class:
                events = self.annotations.get(file_stem, [])
                # Merged events might be better, but assuming raw list is fine or merging again is cheap
                # Reuse merged logic if possible, but self.annotations stores raw. 
                # Let's merge locally for correctness
                if events:
                    events = self._merge_events(events)
                    for t in range(T):
                        abs_t = start_sec + t
                        # Center of the second is abs_t + 0.5, but let's effectively check if the interval [abs_t, abs_t+1) overlaps with event
                        # Consistent with window logic: overlap > 0
                        t_start = float(abs_t)
                        t_end = float(abs_t + 1)
                        
                        for e_start, e_end in events:
                            if t_start < e_end and t_end > e_start:
                                segment_labels[t] = 1.0
                                break
            
            return feature_tensor, video_label, torch.from_numpy(segment_labels)

        return feature_tensor, video_label

    def get_sample_info(self, idx: int) -> Dict:
        return self.samples[idx].copy()

    def get_class_weights(self) -> torch.Tensor:
        labels = [s['label'] for s in self.samples]
        # Calculate manually since utils might not be importable easily or to avoid dependency
        labels = np.array(labels)
        n_samples = len(labels)
        n_classes = 2
        count_0 = (labels == 0).sum()
        count_1 = (labels == 1).sum()
        
        if count_0 == 0 or count_1 == 0:
            return torch.ones(n_classes) # Fallback

        # Standard class weighting: n_samples / (n_classes * count)
        w0 = n_samples / (n_classes * count_0)
        w1 = n_samples / (n_classes * count_1)
        return torch.tensor([w0, w1], dtype=torch.float32)

