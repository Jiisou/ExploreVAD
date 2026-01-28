import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 시간 변환 헬퍼 함수
def time_to_seconds(t_str):
    if pd.isna(t_str) or not isinstance(t_str, str):
        return 0.0
    try:
        h, m, s = map(float, t_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0.0

# 메타데이터 스캔 및 균형 샘플링 함수
def get_balanced_dataset(feature_dir, timestamp_dir, target_total=10000, ratio=1.0):
    # (1) 어노테이션 로드 및 파일명 정규화
    anomaly_map = {}
    csv_files = glob.glob(os.path.join(timestamp_dir, "*.csv"))
    
    print("Step 1: Parsing CSV annotations...")
    for csv_file in csv_files:
        try:
            # 파일명 일치를 위해 확장자 제거 로직 적용
            df = pd.read_csv(csv_file, usecols=['file_name', 'start_time', 'end_time'])
            for _, row in df.iterrows():
                # 'Abuse001_x264.mp4' -> 'Abuse001_x264'
                clean_name = os.path.splitext(str(row['file_name']).strip())[0]
                if clean_name not in anomaly_map:
                    anomaly_map[clean_name] = []
                anomaly_map[clean_name].append((
                    time_to_seconds(row['start_time']), 
                    time_to_seconds(row['end_time'])
                ))
        except Exception as e:
            print(f"  - Skip {os.path.basename(csv_file)} due to format error")

    # (2) 인덱스 정보 수집 (메모리 절약형 스캔)
    meta_abnormal = []
    meta_normal = []
    npy_files = glob.glob(os.path.join(feature_dir, "**/*.npy"), recursive=True)
    
    print(f"Step 2: Scanning {len(npy_files)} feature files...")
    for npy_path in tqdm(npy_files):
        # 'Abuse001_x264.npy' -> 'Abuse001_x264'
        video_name = os.path.splitext(os.path.basename(npy_path))[0]
        
        # 'Normal' 폴더 혹은 어노테이션 미존재 시 전체 정상
        is_video_entirely_normal = "Normal" in npy_path or video_name not in anomaly_map
        
        # 파일 전체를 로드하지 않고 shape만 먼저 확인 (메모리 최적화)
        feat_data = np.load(npy_path, mmap_mode='r')
        num_snippets = feat_data.shape[0]
        
        for i in range(num_snippets):
            # 시간 해석 규칙: (N * 16 // 30)
            timestamp = (i * 16) // 30
            
            label = 'normal'
            if not is_video_entirely_normal:
                if any(start <= timestamp <= end for start, end in anomaly_map[video_name]):
                    label = 'abnormal'
            
            if label == 'abnormal':
                meta_abnormal.append((npy_path, i, 'abnormal'))
            else:
                meta_normal.append((npy_path, i, 'normal'))

    # (3) 비율에 따른 샘플링
    n_abn_target = int(target_total / (1 + ratio))
    n_abn_actual = min(len(meta_abnormal), n_abn_target)
    n_nor_actual = int(n_abn_actual * ratio)
    
    sampled_abn = [meta_abnormal[i] for i in np.random.choice(len(meta_abnormal), n_abn_actual, replace=False)]
    sampled_nor = [meta_normal[i] for i in np.random.choice(len(meta_normal), n_nor_actual, replace=False)]
    
    final_meta = sampled_abn + sampled_nor
    np.random.shuffle(final_meta)
    
    # (4) 실제 피처 데이터 로드
    X, y = [], []
    # 파일별로 그룹화하여 I/O 횟수 최소화
    from collections import defaultdict
    groups = defaultdict(list)
    for p, idx, lab in final_meta: groups[p].append((idx, lab))
    
    print(f"Step 3: Loading {len(final_meta)} sampled features...")
    for p in tqdm(groups.keys()):
        data = np.load(p)
        for idx, lab in groups[p]:
            X.append(data[idx])
            y.append(lab)
            
    return np.array(X), np.array(y)


if __name__ == "__main__":
    FEATURE_DIR = '/mnt/c/JJS/UCF_Crimes/Features/CLIP-ViT-B32/train'
    TIMESTAMP_DIR = '/mnt/c/JJS/UCF_Crimes/Videos/train/00_timestamp'

    # 데이터 준비 (1:1 비율로 10,000개 추출)
    X, y = get_balanced_dataset(FEATURE_DIR, TIMESTAMP_DIR, target_total=10000, ratio=1.0)

    # 차원의 저주(Curse of Dimensionality) 방지를 위해 PCA 선행 (50차원)
    print("Running PCA for dimension reduction...")
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X)

    # t-SNE 실행 (진행 로그 확인을 위해 verbose=1)
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    # 결과 시각화
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette={'abnormal':'red', 'normal':'blue'}, s=10, alpha=0.6)
    plt.title(f"t-SNE Visualization (Samples: {len(y)}, Ratio 1:1)")
    plt.savefig("vad_tsne_result.png")
    plt.show()