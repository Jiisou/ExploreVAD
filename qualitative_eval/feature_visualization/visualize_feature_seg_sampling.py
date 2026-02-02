import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict


def time_to_seconds(t_str):
    """시간 문자열을 초 단위로 변환"""
    if pd.isna(t_str) or not isinstance(t_str, str):
        return 0.0
    try:
        h, m, s = map(float, t_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0.0


def parse_annotations(timestamp_dir):
    """어노테이션 CSV 파일들을 파싱하여 anomaly_map 반환"""
    anomaly_map = {}
    csv_files = glob.glob(os.path.join(timestamp_dir, "*.csv"))

    print("Step 1: Parsing CSV annotations...")
    print(f"  - Found {len(csv_files)} CSV files in {timestamp_dir}")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, usecols=['file_name', 'start_time', 'end_time'])
            for _, row in df.iterrows():
                clean_name = os.path.splitext(str(row['file_name']).strip())[0]
                if clean_name not in anomaly_map:
                    anomaly_map[clean_name] = []
                anomaly_map[clean_name].append((
                    time_to_seconds(row['start_time']),
                    time_to_seconds(row['end_time'])
                ))
        except Exception as e:
            print(f"  - Skip {os.path.basename(csv_file)} due to format error: {e}")

    print(f"  - Parsed {len(anomaly_map)} videos with anomaly annotations")
    if anomaly_map:
        sample_keys = list(anomaly_map.keys())[:3]
        print(f"  - Sample video names in anomaly_map: {sample_keys}")
    return anomaly_map


def scan_metadata(feature_dir, anomaly_map):
    """피처 파일들을 스캔하여 메타데이터 수집 (3가지 카테고리)"""
    meta_abnormal = []
    meta_entirely_normal = []  # Normal 폴더 또는 normal 파일명
    meta_non_abnormal = []     # 이상 영상 내 정상 구간

    npy_files = glob.glob(os.path.join(feature_dir, "**/*.npy"), recursive=True)

    print(f"Step 2: Scanning {len(npy_files)} feature files...")
    for npy_path in tqdm(npy_files):
        video_name = os.path.splitext(os.path.basename(npy_path))[0]

        is_video_entirely_normal = "Normal" in npy_path or "normal" in video_name.lower()
        has_anomaly_annotation = video_name in anomaly_map

        feat_data = np.load(npy_path, mmap_mode='r')
        num_snippets = feat_data.shape[0]

        for i in range(num_snippets):
            timestamp = (i * 16) // 30

            if is_video_entirely_normal or not has_anomaly_annotation:
                meta_entirely_normal.append((npy_path, i, 'entirely_normal'))
            else:
                if any(start <= timestamp <= end for start, end in anomaly_map[video_name]):
                    meta_abnormal.append((npy_path, i, 'abnormal'))
                else:
                    meta_non_abnormal.append((npy_path, i, 'non_abnormal'))

    print(f"  - Found: abnormal={len(meta_abnormal)}, entirely_normal={len(meta_entirely_normal)}, non_abnormal={len(meta_non_abnormal)}")
    return meta_abnormal, meta_entirely_normal, meta_non_abnormal


def sample_binary(meta_abnormal, meta_entirely_normal, meta_non_abnormal, target_total, ratio):
    """Binary 모드: abnormal vs normal (1:ratio)"""
    # normal = entirely_normal + non_abnormal
    meta_normal = [(p, i, 'Normal') for p, i, _ in meta_entirely_normal + meta_non_abnormal]

    n_abn_target = int(target_total / (1 + ratio))
    n_abn_actual = min(len(meta_abnormal), n_abn_target)
    n_nor_actual = min(len(meta_normal), int(n_abn_actual * ratio))

    sampled_abn = [meta_abnormal[i] for i in np.random.choice(len(meta_abnormal), n_abn_actual, replace=False)]
    sampled_nor = [meta_normal[i] for i in np.random.choice(len(meta_normal), n_nor_actual, replace=False)]

    final_meta = sampled_abn + sampled_nor
    np.random.shuffle(final_meta)

    print(f"  - Sampled (binary): abnormal={len(sampled_abn)}, normal={len(sampled_nor)}")
    return final_meta


def sample_ternary(meta_abnormal, meta_entirely_normal, meta_non_abnormal, target_total):
    """Ternary 모드: abnormal vs entirely_normal vs non_abnormal (1:1:1)"""
    n_per_class = target_total // 3

    n_abn = min(len(meta_abnormal), n_per_class)
    n_entirely = min(len(meta_entirely_normal), n_per_class)
    n_non_abn = min(len(meta_non_abnormal), n_per_class)

    sampled_abn = [meta_abnormal[i] for i in np.random.choice(len(meta_abnormal), n_abn, replace=False)]
    sampled_entirely = [meta_entirely_normal[i] for i in np.random.choice(len(meta_entirely_normal), n_entirely, replace=False)] if n_entirely > 0 else []
    sampled_non_abn = [meta_non_abnormal[i] for i in np.random.choice(len(meta_non_abnormal), n_non_abn, replace=False)] if n_non_abn > 0 else []

    final_meta = sampled_abn + sampled_entirely + sampled_non_abn
    np.random.shuffle(final_meta)

    print(f"  - Sampled (ternary 1:1:1): abnormal={len(sampled_abn)}, entirely_normal={len(sampled_entirely)}, non_abnormal={len(sampled_non_abn)}")
    return final_meta


def sample_single_class(meta_list, label, target_total):
    """단일 클래스만 샘플링"""
    n_samples = min(len(meta_list), target_total)
    if n_samples == 0:
        print(f"  - Warning: No samples available for '{label}'")
        return []

    sampled = [meta_list[i] for i in np.random.choice(len(meta_list), n_samples, replace=False)]
    print(f"  - Sampled ({label} only): {len(sampled)}")
    return sampled


def sample_abn_vs_non_abn(meta_abnormal, meta_non_abnormal, target_total):
    """abnormal vs non_abnormal (1:1) - 이상 영상 내 구간만 비교"""
    n_per_class = target_total // 2

    n_abn = min(len(meta_abnormal), n_per_class)
    n_non_abn = min(len(meta_non_abnormal), n_per_class)

    sampled_abn = [meta_abnormal[i] for i in np.random.choice(len(meta_abnormal), n_abn, replace=False)] if n_abn > 0 else []
    sampled_non_abn = [meta_non_abnormal[i] for i in np.random.choice(len(meta_non_abnormal), n_non_abn, replace=False)] if n_non_abn > 0 else []

    final_meta = sampled_abn + sampled_non_abn
    np.random.shuffle(final_meta)

    print(f"  - Sampled (abn_vs_non_abn 1:1): abnormal={len(sampled_abn)}, non_abnormal={len(sampled_non_abn)}")
    return final_meta


def load_features(final_meta):
    """샘플링된 메타데이터로부터 실제 피처 로드"""
    X, y = [], []
    groups = defaultdict(list)
    for p, idx, lab in final_meta:
        groups[p].append((idx, lab))

    print(f"Step 3: Loading {len(final_meta)} sampled features...")
    for p in tqdm(groups.keys()):
        data = np.load(p)
        for idx, lab in groups[p]:
            X.append(data[idx])
            y.append(lab)

    return np.array(X), np.array(y)


def get_balanced_dataset(feature_dir, timestamp_dir, target_total=10000, ratio=1.0, label_mode='binary'):
    """메인 데이터셋 로드 함수"""
    anomaly_map = parse_annotations(timestamp_dir)
    meta_abnormal, meta_entirely_normal, meta_non_abnormal = scan_metadata(feature_dir, anomaly_map)

    if label_mode == 'binary':
        final_meta = sample_binary(meta_abnormal, meta_entirely_normal, meta_non_abnormal, target_total, ratio)
    elif label_mode == 'ternary':
        final_meta = sample_ternary(meta_abnormal, meta_entirely_normal, meta_non_abnormal, target_total)
    elif label_mode == 'abnormal_only':
        final_meta = sample_single_class(meta_abnormal, 'abnormal', target_total)
    elif label_mode == 'non_abnormal_only':
        final_meta = sample_single_class(meta_non_abnormal, 'non_abnormal', target_total)
    elif label_mode == 'entirely_normal_only':
        final_meta = sample_single_class(meta_entirely_normal, 'entirely_normal', target_total)
    elif label_mode == 'abn_vs_non_abn':
        final_meta = sample_abn_vs_non_abn(meta_abnormal, meta_non_abnormal, target_total)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    return load_features(final_meta)


def visualize_2d(X_tsne, y, color_map, labels_order, label_mode):
    """2D t-SNE 시각화"""
    plt.figure(figsize=(10, 7))
    for label in labels_order:
        mask = y == label
        if mask.sum() > 0:
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=color_map[label], label=label, s=10, alpha=0.6)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f"t-SNE 2D ({label_mode}, Samples: {len(y)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"vad_tsne_{label_mode}_2d.png", dpi=300)
    plt.show()


def visualize_3d(X_tsne, y, color_map, labels_order, label_mode):
    """3D t-SNE 시각화"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for label in labels_order:
        mask = y == label
        if mask.sum() > 0:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], X_tsne[mask, 2],
                      c=color_map[label], label=label, s=10, alpha=0.6)

    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title(f"t-SNE 3D ({label_mode}, Samples: {len(y)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"vad_tsne_{label_mode}_3d.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for VAD features')
    parser.add_argument('--feature-dir', type=str, required=True, help='Feature directory path')
    parser.add_argument('--timestamp-dir', type=str, help='Timestamp CSV directory path', default="/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp/")
    parser.add_argument('--label-mode', type=str,
                        choices=['binary', 'ternary', 'abnormal_only', 'non_abnormal_only', 'entirely_normal_only', 'abn_vs_non_abn'],
                        default='binary',
                        help='binary: normal/abnormal, ternary: 1:1:1, abnormal_only, non_abnormal_only, entirely_normal_only, abn_vs_non_abn: abnormal vs non_abnormal (1:1)')
    parser.add_argument('--n-components', type=int, choices=[2, 3], default=2, help='t-SNE dimensions')
    parser.add_argument('--target-total', type=int, default=10000, help='Total samples to extract')
    parser.add_argument('--ratio', type=float, default=1.0, help='normal:abnormal ratio (binary mode only)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 데이터 로드
    X, y = get_balanced_dataset(
        args.feature_dir,
        args.timestamp_dir,
        target_total=args.target_total,
        ratio=args.ratio,
        label_mode=args.label_mode
    )

    # PCA 차원 축소
    print("Running PCA for dimension reduction...")
    X_pca = PCA(n_components=50, random_state=args.seed).fit_transform(X)

    # t-SNE 실행
    print(f"Running t-SNE with {args.n_components}D...")
    tsne = TSNE(n_components=args.n_components, perplexity=30, random_state=args.seed,
                verbose=1, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    # 시각화 설정
    color_map = {
        'abnormal': 'red',
        'Normal': 'blue',
        'entirely_normal': 'blue',
        'non_abnormal': 'green'
    }

    if args.label_mode == 'binary':
        labels_order = ['abnormal', 'Normal']
    elif args.label_mode == 'ternary':
        labels_order = ['abnormal', 'entirely_normal', 'non_abnormal']
    elif args.label_mode == 'abnormal_only':
        labels_order = ['abnormal']
    elif args.label_mode == 'non_abnormal_only':
        labels_order = ['non_abnormal']
    elif args.label_mode == 'entirely_normal_only':
        labels_order = ['entirely_normal']
    elif args.label_mode == 'abn_vs_non_abn':
        labels_order = ['abnormal', 'non_abnormal']
    else:
        labels_order = list(set(y))

    # 시각화
    if args.n_components == 2:
        visualize_2d(X_tsne, y, color_map, labels_order, args.label_mode)
    else:
        visualize_3d(X_tsne, y, color_map, labels_order, args.label_mode)


if __name__ == "__main__":
    main()
