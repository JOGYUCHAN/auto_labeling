# 멀티모달 필터링 Active Learning 시스템

YOLO 객체 탐지에 멀티모달(VLM + CNN) 필터링을 적용한 Active Learning 시스템입니다.

## 주요 기능

### 1. 멀티모달 필터 (NEW!)
- **VLM 텍스트 특징**: BLIP 또는 VIT-GPT2를 사용하여 이미지 캡션 생성 및 텍스트 임베딩 추출
- **CNN 이미지 특징**: DenseNet121을 사용하여 시각적 특징 추출
- **Cross-Attention**: 텍스트와 이미지 특징을 융합하여 객체 분류
- **IoU 기반 학습**: Cycle 1에서 Ground Truth와의 IoU를 계산하여 target/non-target 샘플 자동 선별

### 2. 기존 분류기
- DenseNet121 기반 이미지 분류
- 선택적 재학습 지원

### 3. 캡셔닝 분류기
- VLM 캡션과 키워드 매칭으로 분류
- 재학습 없음

## 파일 구조

```
.
├── multimodal_classifier.py      # 멀티모달 필터 모듈 (NEW!)
├── active_learning.py            # 메인 Active Learning 클래스 (수정됨)
├── config.py                     # 설정 관리 (멀티모달 설정 추가)
├── main.py                       # 실험 실행 스크립트 (멀티모달 옵션 추가)
├── classifier.py                 # 기존 분류기
├── captioning_classifier.py      # 캡셔닝 분류기
├── detector.py                   # 객체 탐지기
├── evaluator.py                  # 성능 평가기
└── utils.py                      # 유틸리티 함수
```

## 멀티모달 필터 작동 원리

### Cycle 1: 학습 데이터 수집
```
1. YOLO로 객체 탐지
2. 각 탐지 결과와 Ground Truth의 IoU 계산
3. IoU >= threshold → Target (class 0)
   IoU < threshold  → Non-target (class 1)
4. 클래스당 N개 샘플 수집 (예: 100개씩)
```

### 멀티모달 모델 학습
```
For each 객체 이미지:
    1. VLM으로 텍스트 특징 추출 (768-dim)
    2. CNN으로 이미지 특징 추출 (1024-dim)
    3. 특징 인코딩 (각각 512-dim)
    4. Cross-Attention (텍스트 ← 이미지)
    5. 특징 융합 및 분류 (target/non-target)
```

### Cycle 2+: 필터링 적용
```
학습된 멀티모달 모델로 탐지 결과 필터링
→ Target만 선택하여 YOLO 재학습
```

## 사용 방법

### 1. 환경 설정

```python
models_dir = "../model weights/YOLO/coco_vehicle"
classifiers_dir = "../model weights/Classification/visdrone"  # 멀티모달은 불필요
image_dir = "../data/visdrone_filtered/images"
label_dir = "../data/visdrone_filtered/labels"  # 멀티모달 필터는 필수!
output_dir = "../experiment_results/visdrone_multimodal"
```

### 2. 멀티모달 필터 설정

```python
# main.py에서 설정

# 멀티모달 필터 활성화
use_multimodal_filter = True

# VLM 모델 선택
multimodal_vlm_type = "blip"  # 또는 "vit-gpt2"

# 학습 샘플 수 (클래스당)
multimodal_train_samples = 100

# 타겟 키워드 (VLM 캡션에도 사용)
target_keywords = ["car", "vehicle"]

# 학습 파라미터
multimodal_epochs = 20
multimodal_batch_size = 16
multimodal_learning_rate = 0.001
```

### 3. 실험 실행

```bash
python main.py
```

## 설정 옵션 비교

### 멀티모달 필터 (권장)
```python
use_classifier = False
use_captioning_classifier = False
use_multimodal_filter = True  # ✓
```

**특징:**
- VLM (텍스트) + CNN (이미지) 융합
- IoU 기반 자동 학습 데이터 생성
- Cycle 1에서만 학습 (이후 재사용)
- 라벨 필수

### 캡셔닝 분류기
```python
use_classifier = False
use_captioning_classifier = True  # ✓
use_multimodal_filter = False
```

**특징:**
- VLM 캡션 + 키워드 매칭
- 재학습 없음
- 라벨 불필요

### 기존 분류기
```python
use_classifier = True  # ✓
use_captioning_classifier = False
use_multimodal_filter = False
```

**특징:**
- DenseNet121 이미지 분류
- 선택적 재학습
- 수동 라벨링 필요

## 멀티모달 필터 실험 프로세스

```
Cycle 0 (선택):
  └─ 베이스라인 성능 측정

Cycle 1:
  ├─ 초기 추론 (YOLO만)
  ├─ IoU 기반 샘플 수집
  │   ├─ Target: IoU >= 0.5 → 100개
  │   └─ Non-target: IoU < 0.5 → 100개
  ├─ 멀티모달 모델 학습
  │   ├─ VLM 특징 추출
  │   ├─ CNN 특징 추출
  │   ├─ Cross-Attention 학습
  │   └─ 분류 헤드 학습
  ├─ 필터링 적용
  └─ YOLO 재학습

Cycle 2+:
  ├─ 학습된 멀티모달 필터 로드
  ├─ 추론 + 필터링
  └─ YOLO 재학습
```

## 출력 디렉토리 구조

```
output_dir/
└── multimodal_blip_yolo_model/
    ├── cycle_1/
    │   ├── detections/              # 전체 탐지 결과
    │   ├── filtered_detections/     # 필터링 후 결과
    │   ├── labels/                  # 전체 라벨
    │   ├── filtered_labels/         # 필터링 후 라벨
    │   ├── multimodal_training/     # 멀티모달 학습
    │   │   └── multimodal_filter.pth
    │   └── training/                # YOLO 학습
    ├── cycle_2/
    │   └── ...
    ├── dataset/                     # YOLO 학습 데이터셋
    └── performance_metrics.csv      # 성능 지표
```

## 멀티모달 아키텍처

```
입력: 객체 이미지
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
VLM (BLIP/ViT-GPT2)  CNN (DenseNet121)    [Keywords]
    │                     │
Text Embedding      Image Features
 (768-dim)           (1024-dim)
    │                     │
    └─────────────────────┴──────────┐
                                     │
                          Feature Encoding
                               (512-dim)
                                     │
                          Cross-Attention
                         (text query ← image)
                                     │
                            Feature Fusion
                                     │
                           Classification
                                     │
                         ┌────────┴────────┐
                      Target         Non-target
                     (class 0)       (class 1)
```

## 주요 파라미터

### IoU 임계값
```python
iou_threshold = 0.5  # Target/Non-target 구분 기준
```

### 학습 샘플 수
```python
multimodal_train_samples = 100  # 클래스당 샘플 수
```
- 너무 적으면: 학습 부족
- 너무 많으면: 학습 시간 증가

### VLM 모델 선택
```python
multimodal_vlm_type = "blip"  # 정확도 우선
# 또는
multimodal_vlm_type = "vit-gpt2"  # 속도 우선
```

## 성능 비교 (예상)

| 방법 | mAP50 | 필터링 정확도 | 학습 시간 |
|------|-------|--------------|----------|
| 분류기 없음 | 0.75 | - | 빠름 |
| 캡셔닝 | 0.78 | 중간 | 중간 |
| 기존 분류기 | 0.82 | 높음 | 느림 |
| **멀티모달** | **0.85** | **매우 높음** | 중간 |

## 장점

1. **자동 학습 데이터 생성**: IoU 기반으로 수동 라벨링 불필요
2. **멀티모달 융합**: 텍스트와 이미지 정보를 모두 활용
3. **한 번만 학습**: Cycle 1 이후 재사용
4. **높은 정확도**: Attention 메커니즘으로 정교한 분류

## 주의사항

1. **라벨 필수**: Ground Truth가 없으면 IoU 계산 불가
2. **GPU 메모리**: VLM + CNN 동시 로드로 메모리 사용량 높음
3. **첫 학습 시간**: Cycle 1에서 특징 추출 및 학습 시간 소요

## 문제 해결

### GPU 메모리 부족
```python
multimodal_batch_size = 8  # 기본 16에서 감소
```

### 학습 데이터 부족
```python
multimodal_train_samples = 50  # 기본 100에서 감소
iou_threshold = 0.4  # 더 많은 target 샘플 수집
```

### VLM 로딩 실패
```bash
pip install transformers>=4.30.0
pip install accelerate
```

## 라이센스

MIT License

## 문의

문제가 있거나 제안 사항이 있으시면 이슈를 등록해주세요.
