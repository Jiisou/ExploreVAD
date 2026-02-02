"""
Core computation module for Vision-Language Similarity Dashboard.

Handles directory scanning, video sampling, text prompt construction,
and cosine similarity computation. No UI dependencies.
"""

import os
import re
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path


def detect_directory_structure(root_dir):
    """
    Detect the directory structure type.

    Returns:
        'split' if has train/test folders, 'flat' if classes are directly in root.
    """
    root = Path(root_dir)
    subdirs = [d.name for d in root.iterdir() if d.is_dir()]
    if "train" in subdirs or "test" in subdirs:
        return "split"
    return "flat"


def discover_classes(root_dir):
    """
    Scan root_dir for class subdirectories containing .npy files.

    Handles both flat and split directory structures:
    - Flat: root/{ClassName}/*.npy
    - Split: root/{train|test}/{ClassName}/*.npy

    Returns:
        sorted list of class names
    """
    root = Path(root_dir)
    structure = detect_directory_structure(root_dir)

    class_names = set()
    if structure == "split":
        for split_name in ["train", "test"]:
            split_dir = root / split_name
            if split_dir.exists():
                for d in split_dir.iterdir():
                    if d.is_dir() and any(d.glob("*.npy")):
                        class_names.add(d.name)
    else:
        for d in root.iterdir():
            if d.is_dir() and any(d.glob("*.npy")):
                class_names.add(d.name)

    return sorted(class_names)


def list_videos_per_class(root_dir):
    """
    List all npy files grouped by class.

    Returns:
        dict[str, list[str]]: {class_name: [list of npy file paths]}
    """
    root = Path(root_dir)
    structure = detect_directory_structure(root_dir)

    videos = {}

    if structure == "split":
        for split_name in ["train", "test"]:
            split_dir = root / split_name
            if not split_dir.exists():
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                npy_files = sorted(class_dir.glob("*.npy"))
                if npy_files:
                    class_name = class_dir.name
                    if class_name not in videos:
                        videos[class_name] = []
                    videos[class_name].extend([str(f) for f in npy_files])
    else:
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            npy_files = sorted(class_dir.glob("*.npy"))
            if npy_files:
                videos[class_dir.name] = [str(f) for f in npy_files]

    return videos


def sample_one_per_class(videos_per_class, seed=42):
    """
    Randomly pick one video npy file per class.

    Args:
        videos_per_class: dict[str, list[str]]
        seed: random seed for reproducibility

    Returns:
        dict[str, str]: {class_name: npy_path}
    """
    rng = random.Random(seed)
    sampled = {}
    for class_name in sorted(videos_per_class.keys()):
        files = videos_per_class[class_name]
        if files:
            sampled[class_name] = rng.choice(files)
    return sampled


def load_video_features(npy_path):
    """
    Load and validate a video feature npy file.

    Returns:
        numpy array of shape [T, D] where T=temporal length, D=embedding dim.
    """
    features = np.load(npy_path).astype(np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return features


def compute_similarity_matrix(video_features, text_embeddings):
    """
    Compute cosine similarity between each frame and each text embedding.

    Since both inputs are L2-normalized, cosine similarity = dot product.

    Args:
        video_features: [T, D] numpy array (L2-normalized)
        text_embeddings: [C, D] numpy array (L2-normalized)

    Returns:
        [T, C] similarity matrix
    """
    return video_features @ text_embeddings.T


def split_camel_case(name):
    """
    Split CamelCase or underscore-separated names into lowercase words.

    Examples:
        'RoadAccidents' -> 'road accidents'
        'Stealing' -> 'stealing'
        'Normal_Videos' -> 'normal videos'
    """
    # Replace underscores with spaces
    name = name.replace("_", " ")
    # Insert space before uppercase letters that follow lowercase
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Insert space between consecutive uppercase and following lowercase
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.lower().strip()


def build_text_prompts(class_names, template="{}"):
    """
    Convert class folder names into text prompts.

    Args:
        class_names: list of str (folder names)
        template: format string with {} placeholder
            - "{}": raw class name (after CamelCase splitting)
            - "a video of {}": e.g., "a video of road accidents"
            - "a photo of {}": CLIP's original zero-shot template

    Returns:
        list of str (same order as class_names)
    """
    prompts = []
    for name in class_names:
        readable = split_camel_case(name)
        prompts.append(template.format(readable))
    return prompts


def time_to_seconds(t_str):
    """
    Convert time string (H:MM:SS or HH:MM:SS) to seconds.

    Args:
        t_str: time string, e.g. "0:00:06" or "0:01:30"

    Returns:
        float seconds
    """
    if pd.isna(t_str) or not isinstance(t_str, str):
        return 0.0
    try:
        parts = list(map(float, t_str.strip().split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            return float(parts[0])
    except Exception:
        return 0.0


def parse_annotations(timestamp_dir):
    """
    Parse all timestamp CSV files from a directory into an anomaly map.

    Each CSV is expected to have columns: file_name, start_time, end_time
    with times in H:MM:SS format.

    Args:
        timestamp_dir: path to directory containing *_timestamps.csv files

    Returns:
        dict[str, list[tuple[float, float]]]:
            {video_base_name: [(start_sec, end_sec), ...]}
    """
    anomaly_map = {}
    csv_files = glob.glob(os.path.join(timestamp_dir, "*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Normalize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]

            if "file_name" not in df.columns:
                continue

            for _, row in df.iterrows():
                fname = str(row["file_name"]).strip()
                # Remove .mp4 extension
                clean_name = os.path.splitext(fname)[0]

                start = time_to_seconds(str(row.get("start_time", "")))
                end = time_to_seconds(str(row.get("end_time", "")))

                if end <= start:
                    continue

                if clean_name not in anomaly_map:
                    anomaly_map[clean_name] = []
                anomaly_map[clean_name].append((start, end))
        except Exception:
            continue

    return anomaly_map


def lookup_annotation(video_name, anomaly_map):
    """
    Look up anomaly intervals for a video name.

    Args:
        video_name: base name of npy file (without extension)
        anomaly_map: dict from parse_annotations()

    Returns:
        list of (start_sec, end_sec) tuples, or empty list if not found.
    """
    return anomaly_map.get(video_name, [])


def apply_smoothing(data, window_size):
    """
    Apply moving average smoothing along the temporal axis.

    Args:
        data: [T, C] numpy array
        window_size: int, kernel size for moving average

    Returns:
        [T, C] smoothed array
    """
    if window_size <= 1:
        return data
    kernel = np.ones(window_size) / window_size
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=data
    )
    return smoothed
