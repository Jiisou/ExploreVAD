# 비디오 레벨 피처 시각화
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse


def aggregate_temporal(feat, method='mean'):
    """
    Aggregate temporal snippets into a single video embedding.

    Args:
        feat: numpy array - can be (T, F) or higher dimensional like (T, C, H, W)
        method: aggregation method
            - 'mean': average pooling (default)
            - 'max': max pooling
            - 'mean_max': concatenate mean and max [2F]
            - 'stats': concatenate mean, max, std [3F]
            - 'first': first snippet
            - 'last': last snippet
            - 'middle': middle snippet
    Returns:
        numpy array of shape (F,) or (2F,) or (3F,) depending on method
    """
    # Flatten all dimensions except temporal (first dim)
    if len(feat.shape) > 2:
        feat = feat.reshape(feat.shape[0], -1)

    if method == 'mean':
        return feat.mean(axis=0)
    elif method == 'max':
        return feat.max(axis=0)
    elif method == 'mean_max':
        return np.concatenate([feat.mean(axis=0), feat.max(axis=0)])
    elif method == 'stats':
        return np.concatenate([feat.mean(axis=0), feat.max(axis=0), feat.std(axis=0)])
    elif method == 'first':
        return feat[0]
    elif method == 'last':
        return feat[-1]
    elif method == 'middle':
        return feat[len(feat) // 2]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def detect_directory_structure(feature_dir):
    """
    Detect the directory structure type.
    Returns: 'split' if has train/test folders, 'flat' if classes are directly in root
    """
    feature_dir = Path(feature_dir)
    subdirs = [d.name for d in feature_dir.iterdir() if d.is_dir()]

    if 'train' in subdirs or 'test' in subdirs:
        return 'split'
    else:
        return 'flat'


def load_features_split(feature_dir, split='train', agg_method='mean'):
    """Load features from train/test split structure."""
    split_dir = Path(feature_dir) / split
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(split_dir / d)])

    features = []
    labels = []
    filenames = []

    for class_name in tqdm(classes, desc=f"Loading {split} features"):
        class_dir = split_dir / class_name
        for npy_file in sorted(class_dir.glob("*.npy")):
            feat = np.load(npy_file).astype(np.float32)
            feat_agg = aggregate_temporal(feat, method=agg_method)
            features.append(feat_agg)
            labels.append(class_name)
            filenames.append(npy_file.name)

    return np.array(features), labels, classes, filenames


def load_features_flat(feature_dir, split='all', agg_method='mean'):
    """
    Load features from flat structure (classes directly in root).
    Normal videos are in 'Training_Normal_Videos_Anomaly' or 'Testing_Normal_Videos_Anomaly' or 'Normal.
    """
    feature_dir = Path(feature_dir)
    all_dirs = sorted([d.name for d in feature_dir.iterdir() if d.is_dir()])

    # Identify normal and anomaly folders
    normal_general = 'Normal'
    normal_train = 'Training_Normal_Videos_Anomaly'
    normal_test = 'Testing_Normal_Videos_Anomaly'
    anomaly_classes = [d for d in all_dirs if d not in [normal_general, normal_train, normal_test]]

    features = []
    labels = []
    filenames = []

    # Load anomaly classes
    for class_name in tqdm(anomaly_classes, desc="Loading anomaly features"):
        class_dir = feature_dir / class_name
        for npy_file in sorted(class_dir.glob("*.npy")):
            feat = np.load(npy_file).astype(np.float32)
            feat_agg = aggregate_temporal(feat, method=agg_method)
            features.append(feat_agg)
            labels.append(class_name)
            filenames.append(npy_file.name)

    # Load normal videos
    normal_dirs = []
    if split in ['all', 'train'] and (feature_dir / normal_train).exists():
        normal_dirs.append(normal_train)
    if split in ['all', 'test'] and (feature_dir / normal_test).exists():
        normal_dirs.append(normal_test)
    if (feature_dir / normal_general).exists():
        normal_dirs.append(normal_general)

    for normal_dir in tqdm(normal_dirs, desc="Loading normal features"):
        class_dir = feature_dir / normal_dir
        for npy_file in sorted(class_dir.glob("*.npy")):
            feat = np.load(npy_file).astype(np.float32)
            feat_agg = aggregate_temporal(feat, method=agg_method)
            features.append(feat_agg)
            labels.append('normal')
            filenames.append(npy_file.name)

    classes = anomaly_classes + ['normal']
    return np.array(features), labels, classes, filenames


def load_features(feature_dir, split='train', agg_method='mean'):
    """Load features with automatic structure detection."""
    structure = detect_directory_structure(feature_dir)
    print(f"Detected directory structure: {structure}")

    if structure == 'split':
        return load_features_split(feature_dir, split, agg_method)
    else:
        return load_features_flat(feature_dir, split, agg_method)


def visualize_tsne(features, labels, classes, n_components=2, title="Feature Visualization", save_path=None):
    """Visualize features using t-SNE (2D or 3D)."""
    print(f"Running t-SNE ({n_components}D) on {len(features)} samples...")
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, max_iter=1000)
    features_reduced = tsne.fit_transform(features)

    # Create color map
    anomaly_classes = [c for c in classes if c != 'normal']
    cmap = plt.colormaps.get_cmap('tab20')
    color_map = {c: cmap(i / max(len(anomaly_classes), 1)) for i, c in enumerate(anomaly_classes)}
    color_map['normal'] = (0.7, 0.7, 0.7, 1.0)

    fig = plt.figure(figsize=(14, 10))

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for class_name in ['normal'] + anomaly_classes:
            mask = np.array([l == class_name for l in labels])
            if mask.sum() > 0:
                ax.scatter(
                    features_reduced[mask, 0],
                    features_reduced[mask, 1],
                    features_reduced[mask, 2],
                    c=[color_map[class_name]],
                    label=f"{class_name} ({mask.sum()})",
                    alpha=0.6 if class_name == 'normal' else 0.8,
                    s=30 if class_name == 'normal' else 50,
                    edgecolors='none'
                )
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
    else:
        ax = fig.add_subplot(111)
        for class_name in ['normal'] + anomaly_classes:
            mask = np.array([l == class_name for l in labels])
            if mask.sum() > 0:
                ax.scatter(
                    features_reduced[mask, 0],
                    features_reduced[mask, 1],
                    c=[color_map[class_name]],
                    label=f"{class_name} ({mask.sum()})",
                    alpha=0.6 if class_name == 'normal' else 0.8,
                    s=30 if class_name == 'normal' else 50,
                    edgecolors='none'
                )
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def visualize_tsne_binary(features, labels, n_components=2, title="Feature Visualization (Binary)", save_path=None):
    """Visualize features using t-SNE with binary labels (normal vs anomaly), 2D or 3D."""
    print(f"Running t-SNE ({n_components}D) on {len(features)} samples...")
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, max_iter=1000)
    features_reduced = tsne.fit_transform(features)

    binary_labels = ['Normal' if l == 'normal' else 'Anomaly' for l in labels]

    fig = plt.figure(figsize=(10, 8))

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for label, color in [('Normal', 'blue'), ('Anomaly', 'red')]:
            mask = np.array([l == label for l in binary_labels])
            ax.scatter(
                features_reduced[mask, 0],
                features_reduced[mask, 1],
                features_reduced[mask, 2],
                c=color,
                label=f"{label} ({mask.sum()})",
                alpha=0.6,
                s=40,
                edgecolors='none'
            )
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
    else:
        ax = fig.add_subplot(111)
        for label, color in [('Normal', 'blue'), ('Anomaly', 'red')]:
            mask = np.array([l == label for l in binary_labels])
            ax.scatter(
                features_reduced[mask, 0],
                features_reduced[mask, 1],
                c=color,
                label=f"{label} ({mask.sum()})",
                alpha=0.6,
                s=40,
                edgecolors='none'
            )
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize video features using t-SNE")
    parser.add_argument("--feature_dir", type=str, default="/mnt/c/JJS/UCF_Crimes/Features/CLIP-TSA",
                        help="Path to feature directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"],
                        help="Which split to visualize (use 'all' for flat structure)")
    parser.add_argument("--mode", type=str, default="multi", choices=["multi", "binary"],
                        help="multi: all classes, binary: normal vs anomaly")
    parser.add_argument("--n_components", type=int, default=2, choices=[2, 3],
                        help="Number of t-SNE dimensions (2 or 3)")
    parser.add_argument("--agg", type=str, default="mean",
                        choices=["mean", "max", "mean_max", "stats", "first", "last", "middle"],
                        help="Temporal aggregation method")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the figure")
    args = parser.parse_args()

    # Load features
    features, labels, classes, filenames = load_features(args.feature_dir, args.split, args.agg)
    print(f"Loaded {len(features)} samples, {len(classes)} classes")
    print(f"Classes: {classes}")
    print(f"Feature shape: {features.shape}")

    # Visualize
    feature_name = Path(args.feature_dir).name
    title = f"{feature_name} Features ({args.split}) - {args.n_components}D - agg:{args.agg}"
    save_path = args.save or f"tsne_{feature_name}_{args.split}_{args.mode}_{args.n_components}d_{args.agg}.png"

    if args.mode == "multi":
        visualize_tsne(features, labels, classes, n_components=args.n_components, title=title, save_path=save_path)
    else:
        visualize_tsne_binary(features, labels, n_components=args.n_components, title=title + " - Binary", save_path=save_path)
