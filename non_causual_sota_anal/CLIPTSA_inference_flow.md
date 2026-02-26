# CLIP-TSA 추론 Flow 및 Frame-level AUC 계산

## 핵심 요약

**Weakly Supervised Video Anomaly Detection Pipeline:**
- 입력: 비디오 레벨 라벨만 있음 (정상/이상)
- 모델: CLIP 특징 + TSA (Hard Attention) + MIL 학습
- 출력: Frame-level 이상도 점수 + AUC 메트릭

## 주요 데이터 흐름

### 입력 단계
1. `.npy` 파일: (T, 10, F) 또는 (T, F) 로드
   - T: 가변 길이 (152~512 프레임)
   - 10: 10-crop augmentation (ShanghaiTech만)
   - F: 2048 (I3D/C3D) 또는 512 (ViT)

2. Segment 화: T → 32 (고정)
   - `np.linspace(0, len(feat), 33)` 이용 균등 분할
   - 각 구간의 평균값을 segment feature로 사용

3. 배치 형태: (1, ncrops, 32, F)
   - 배치 크기 = 1 (테스트시)
   - ncrops = 10 (ShanghaiTech) 또는 1 (ViT)

### 모델 처리
1. MLP 축소: F=2048 → 512 (I3D/C3D만)
2. Hard Attention (TSA): 32 segment → 30 segment 선택
   - PerturbedTopK: Gaussian perturbation + top-k 선택
   - k = int(32 * 0.95) = 30
3. Aggregate: Multi-scale Conv1D + Non-Local Block
4. Classification: FC layers → (1, 32, 1) 출력

### Frame-level 변환
- Segment 점수 평균화: 비디오 점수 1개
- `np.repeat(pred, 16)`: 각 점수를 16번 반복
  - 근거: 32 segment / 512 프레임 ≈ 16 프레임/segment

### AUC 계산
- ROC-AUC: `sklearn.metrics.roc_curve()` + `auc()`
- Precision-Recall AUC: XD-Violence만 (폭력 dataset)

## 중요 파일 위치
- 추론: `test_10crop.py:8-54`
- 모델: `model.py:195-327`
- 데이터: `dataset.py:88-127` + `utils/utils.py:4-24`
- Hard Attention: `utils/hard_attention.py:85-104`
- 손실함수: `train.py:50-127`

## Weakly Supervised 핵심
- 비디오 레벨 라벨만으로 학습
- MIL: 정상 비디오는 모든 segment 정상, 이상은 최소 1개 이상
- Hard Attention: 모델이 이상 segment를 자동으로 발견
