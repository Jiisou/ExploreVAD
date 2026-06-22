# CLIPVAD 추론 흐름 (Weakly-Supervised Learning)

## 개요
비디오 클래스 라벨(Normal, Abuse, Arrest 등)만으로 학습한 모델이 추론 시 **프레임 단위 이상 탐지**를 수행하는 방식을 설명합니다.

---

## 1. Weakly-Supervised Learning Context

### 학습 데이터 구성
- **라벨 단위**: 비디오 클래스 (전체 비디오 1개 = 1개 라벨)
- **예시**: "video_001.npy" → "Assault" (비디오 전체가 폭력 범죄)
- **프레임 단위 라벨**: 없음 (약한 감독)

### 모델 학습 목표
- 비디오 전체의 클래스 정보만으로 **프레임 단위 판별 능력**을 간접적으로 학습
- 예: "Assault" 비디오 내 모든 프레임들을 보며 "이들이 폭력 장면인가?"를 학습

---

## 2. 추론 흐름 (입력 → 출력)

### 2.1 입력 (Dataset)

```
UCFDataset.__getitem__(index)
    ↓
clip_feature = np.load(path)  # shape: (num_clips, 512) [CLIP feature]
                               # num_clips: 비디오의 클립 수 (가변)
```

**데이터 예시**:
```
비디오: UCF_101_video.npy
- 원본 클립 수: 1000개
- 각 클립 차원: 512 (ViT-B/16 CLIP 임베딩)
- 각 클립 = 16개 프레임의 평균 풀링 결과
```

#### process_split 함수 (슬라이딩 윈도우)

```python
clip_feature, clip_length = tools.process_split(clip_feature, maxlen=256)
```

**동작 원리**:
- `maxlen=256`: 모델 입력 크기 (최대 256개 클립 = 4096 프레임)
- 입력 클립 수가 256 초과 → **비중첩 슬라이딩 윈도우**로 분할

**예시** (1000개 클립):
```
원본 클립: [0, 1, 2, ..., 999]
           ↓ process_split(maxlen=256)
윈도우 0: [0~255]    (256개 클립)
윈도우 1: [256~511]  (256개 클립)
윈도우 2: [512~767]  (256개 클립)
윈도우 3: [768~999]  (232개 클립, 패딩됨)

출력 shape: (4, 256, 512)  # 4개 윈도우 × 256개 클립 × 512차원
실제 clip_length: 1000
```

### 2.2 모델 순전파 (Forward Pass)

```python
visual = visual.unsqueeze(0) if len_cur < maxlen  # shape: (1, 256, 512)
visual = visual.to(device)

lengths = [256, 256, 256, 232]  # 각 윈도우의 실제 클립 수
padding_mask = get_batch_mask(lengths, maxlen)
    # shape: (4, 256) - padding된 위치 표시

_, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
```

#### CLIPVAD 모델 내부 처리

```
입력: (num_windows=4, maxlen=256, feat_dim=512)
      |
      ├─ encode_video()
      │   ├─ Position Embedding 추가 (frame_position_embeddings)
      │   ├─ Temporal Transformer (길이 기반 윈도우 주의)
      │   │   - attn_window=8: 8개 클립씩 주의 범위 제한
      │   │   - 계산 효율성을 위한 로컬 주의
      │   ├─ Graph Convolution (GCN) × 4개 레이어
      │   │   ├─ gc1, gc2: 유사도 기반 그래프
      │   │   └─ gc3, gc4: 거리 기반 그래프
      │   └─ Fusion & Linear
      │
      ├─ encode_textprompt(text)
      │   └─ CLIP 텍스트 인코더 (frozen)
      │       입력: ["normal", "abuse", "arrest", ...]
      │       출력: (14, 512) # 14개 클래스 임베딩
      │
      └─ 분류 헤드
          ├─ logits1 = classifier(visual_features)  # (4, 256, 1)
          │             ↑ 이상 탐지 점수 (binary sigmoid)
          │
          └─ logits2 = visual_features @ text_features  # (4, 256, 14)
                       ↑ 텍스트-시각 유사도 (14개 클래스)

출력 shape:
- logits1: (4, 256, 1)    # 이상 점수 (각 클립)
- logits2: (4, 256, 14)   # 클래스 확률 (각 클립)
```

### 2.3 후처리 (Post-processing)

```python
# 1. 윈도우 구조 제거 (reshape)
logits1_flat = logits1.reshape(4*256, 1)  # (1024, 1)
logits2_flat = logits2.reshape(4*256, 14)  # (1024, 14)

# 2. 실제 클립만 추출
logits1_valid = logits1_flat[0:len_cur]  # (1000, 1)
logits2_valid = logits2_flat[0:len_cur]  # (1000, 14)

# 3. 확률로 변환
prob1 = sigmoid(logits1_valid)       # (1000,) ∈ [0, 1]
prob2 = softmax(logits2_valid, dim=1)  # (1000, 14)
```

**각 변수의 의미**:
```
prob1[i] ∈ [0, 1]
  = 클립 i가 이상인 확률
  = 실수값 점수 (이후 frame-level로 확장됨)

prob2[i, j] ∈ [0, 1]
  = 클립 i가 클래스 j일 확률
  = 14개 클래스 확률 분포
```

---

## 3. Frame-Level AUC 측정 방법

### 3.1 Ground Truth 준비

```python
gt = np.load('list/gt_ucf.npy')  # shape: (총_프레임_수,)
                                  # 각 요소: 0 (정상) 또는 1 (이상)

# 예: [0, 0, 0, 1, 1, 1, 0, 0, ...]
#      ↑ video_1의 프레임들  ↑ video_2의 프레임들...
```

**gt 배열 구성**:
```
Video 1 (1000 클립 = 16,000 프레임):
  정상 구간: frame [0~8000)    → gt = 0
  이상 구간: frame [8000~12000) → gt = 1
  정상 구간: frame [12000~16000) → gt = 0

Video 2 (500 클립 = 8,000 프레임):
  이상 구간: frame [16000~24000) → gt = 1

...
```

### 3.2 Clip-Level → Frame-Level 확장

```python
# 각 클립 = 16개 프레임
# clip_i의 확률 → frame [i*16 : (i+1)*16] 모두에 복제

prob1_clipped = prob1  # (1000,) - clip-level
prob1_frame = np.repeat(prob1_clipped, 16)  # (16000,) - frame-level

# 예시:
prob1_clipped = [0.1, 0.2, 0.8, 0.9]  # 4개 클립
prob1_frame = [0.1, 0.1, ..., 0.1,    # 클립 0의 16개 프레임
               0.2, 0.2, ..., 0.2,    # 클립 1의 16개 프레임
               0.8, 0.8, ..., 0.8,    # 클립 2의 16개 프레임
               0.9, 0.9, ..., 0.9]    # 클립 3의 16개 프레임
# shape: (64,)
```

### 3.3 AUC 계산

```python
from sklearn.metrics import roc_auc_score

# Frame-level 비교
AUC = roc_auc_score(gt, prob1_frame)

# 예:
# gt      = [0, 0, ..., 0, 1, 1, ..., 1, 0, ...]
# prob1_frame = [0.1, 0.1, ..., 0.15, 0.85, ..., 0.9, 0.2, ...]
# AUC     = 0.92
```

#### AUC 해석
```
ROC Curve:
- X축: False Positive Rate (정상을 이상으로 오분류)
- Y축: True Positive Rate (이상을 이상으로 올바르게 분류)

AUC = 0.92 의미:
  무작위로 선택한 {정상 프레임, 이상 프레임} 쌍에 대해
  92% 확률로 이상 프레임이 더 높은 점수를 갖음
```

### 3.4 AP (Average Precision) 계산

```python
from sklearn.metrics import average_precision_score

AP = average_precision_score(gt, prob1_frame)

# Precision-Recall 곡선 아래 면적
# 특히 이상 탐지에서 불균형 데이터에 유용
# (정상 >> 이상 인 경우)
```

---

## 4. 전체 흐름 다이어그램

```
[ 입력 ]
   비디오.npy (1000개 CLIP 임베딩)
        |
        ↓
[ 전처리: Sliding Window ]
   process_split(maxlen=256)
        |
        ├─ Window 0: CLIP[0~255]     (256개 클립)
        ├─ Window 1: CLIP[256~511]   (256개 클립)
        ├─ Window 2: CLIP[512~767]   (256개 클립)
        └─ Window 3: CLIP[768~999]   (232 + 24 패딩)
        |
        ↓
[ 모델 순전파 ]
   shape: (4 windows, 256 maxlen, 512 feat)
        |
        ├─ encode_video() → (4, 256, 256)
        │   └─ Temporal + GCN 처리
        │
        └─ classifier() → logits1: (4, 256, 1)
        |                logits2: (4, 256, 14)
        |
        ↓
[ 후처리 ]
   reshape → (1024, 1), (1024, 14)
   유효 클립만 추출 → (1000, 1), (1000, 14)
   sigmoid/softmax → prob1: (1000,), prob2: (1000, 14)
        |
        ↓
[ Frame-Level 확장 ]
   np.repeat(prob1, 16) → (16000,)
        |
        ↓
[ 평가 ]
   gt_frame: (16000,) - ground truth
   pred_frame: (16000,) - prediction
        |
        ├─ AUC = roc_auc_score(gt_frame, pred_frame)
        └─ AP = average_precision_score(gt_frame, pred_frame)
```

---

## 5. 코드 실행 예시 (`ucf_test.py`)

```python
# Line 23-31: 데이터 로딩 및 전처리
visual = item[0].squeeze(0)      # (1000, 512) 또는 (num_windows, 256, 512)
length = int(item[2])            # 1000 (총 클립 수)
len_cur = length

if len_cur < maxlen:
    visual = visual.unsqueeze(0) # (1, 256, 512)

# Line 33-45: window lengths 계산
lengths = torch.zeros(int(length / maxlen) + 1)  # [256, 256, 256, 232]
for j in range(int(length / maxlen) + 1):
    # lengths 배열 채우기

# Line 47: 모델 순전파
_, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
# logits1: (4, 256, 1)
# logits2: (4, 256, 14)

# Line 48-51: reshape 및 sigmoid
logits1_flat = logits1.reshape(1024, 1)
prob1 = torch.sigmoid(logits1_flat[0:len_cur].squeeze(-1))  # (1000,)

# Line 70-71: Frame-level AUC/AP
ROC1 = roc_auc_score(gt, np.repeat(prob1, 16))
AP1 = average_precision_score(gt, np.repeat(prob1, 16))

print(f"AUC: {ROC1:.4f}")  # Example: 0.9234
print(f"AP:  {AP1:.4f}")   # Example: 0.8876
```

---

## 6. 주요 특징 정리

| 항목 | 값 | 설명 |
|------|-----|------|
| **학습 라벨** | 비디오 클래스 | Weakly-supervised |
| **입력 단위** | CLIP 임베딩 | 1개 = 16 프레임의 평균 |
| **윈도우 크기** | 256개 클립 | 4,096 프레임 = ~273초 (15FPS) |
| **모델 출력** | clip-level 점수 | (num_windows, 256, 1) |
| **평가 단위** | frame-level | (총_프레임_수,) |
| **확장 비율** | ×16 | np.repeat(clip_pred, 16) |
| **평가 지표** | AUC, AP | ROC곡선, Precision-Recall |

---

## 7. 모델 전체 파라미터 개수

### UCF-Crime 구성 (default args)

```
Model Configuration:
  - embed_dim: 512
  - visual_length: 256
  - visual_width: 512
  - visual_head: 1
  - visual_layers: 2
  - classes_num: 14

Trainable Parameters:
  - Temporal Transformer: ~2.1M
  - Graph Convolution (4×): ~1.0M
  - Classifier & MLPs: ~0.3M
  - Position/Prompt Embeddings: ~0.1M
  ─────────────────────────────
  Total Trainable: ~3.5M

Frozen Parameters (CLIP ViT-B/16):
  - CLIP Model: ~86.6M
  ─────────────────────────────
  Total: ~90M parameters
```

---

## 8. Weakly-Supervised 학습의 핵심

### 학습 시
```
비디오 라벨: "Assault"
        ↓
전체 비디오의 모든 프레임 임베딩
        ↓
CLIPVAD 모델: "이 임베딩들의 집합이 Assault인가?"
        ↓
Loss: BCE(pred_logits, video_label)
```

### 추론 시
```
학습과 동일한 모델 구조
        ↓
각 클립의 독립적 이상 점수 생성
        ↓
Frame-level 평가 (프레임 GT와 비교)
```

**핵심**: 비디오 전체로만 학습했지만, **프레임 단위의 판별 능력을 간접적으로 획득**
