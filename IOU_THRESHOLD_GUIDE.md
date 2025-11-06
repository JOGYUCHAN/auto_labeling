# IoU 임계값 설정 가이드

## 📊 개요

멀티모달 필터는 **Cycle 1**에서 IoU(Intersection over Union)를 기준으로 Target/Non-target 샘플을 자동 분류합니다.

---

## ⚙️ main.py 설정

```python
# main.py의 멀티모달 설정 섹션

# 멀티모달 Target/Non-target 분류 IoU 임계값
multimodal_iou_threshold = 0.5  # 기본값 (권장)

# 권장값:
# - 0.3: 매우 관대 (더 많은 Target 샘플)
# - 0.4: 관대 (충분한 Target 샘플)
# - 0.5: 표준 (COCO 기준, 권장) ✓
# - 0.6: 엄격 (고품질 Target만)
# - 0.7: 매우 엄격 (최고 품질만, Target 부족 가능)
```

---

## 🎯 분류 규칙

### 기본 규칙 (임계값 = 0.5)

```python
if IoU >= 0.5:
    → Target (class 0)
    → "올바른 탐지, 유지해야 함"
else:
    → Non-target (class 1)
    → "잘못된 탐지, 제거해야 함"
```

### 예시

| IoU 값 | 기준 0.3 | 기준 0.5 | 기준 0.7 |
|--------|---------|---------|---------|
| 0.85 | ✅ Target | ✅ Target | ✅ Target |
| 0.65 | ✅ Target | ✅ Target | ❌ Non-target |
| 0.55 | ✅ Target | ✅ Target | ❌ Non-target |
| 0.45 | ✅ Target | ❌ Non-target | ❌ Non-target |
| 0.35 | ✅ Target | ❌ Non-target | ❌ Non-target |
| 0.25 | ❌ Non-target | ❌ Non-target | ❌ Non-target |

---

## 📐 임계값별 특성

### 0.3 (매우 관대)
```
장점:
  ✓ Target 샘플 매우 많음
  ✓ 학습 데이터 수집 쉬움
  ✓ 부분 탐지도 Target으로 인정

단점:
  ✗ 부정확한 탐지 포함
  ✗ 모델이 낮은 품질 학습
  ✗ 필터링 성능 저하

사용 시기:
  - Target 샘플 수집이 매우 어려운 경우
  - 초기 실험/테스트
```

### 0.4 (관대)
```
장점:
  ✓ 충분한 Target 샘플
  ✓ 적당히 관대한 기준
  ✓ 학습 데이터 수집 용이

단점:
  ✗ 경계선 케이스 포함
  ✗ 약간 낮은 품질

사용 시기:
  - Target 샘플 부족한 경우
  - YOLO 초기 성능이 낮은 경우
```

### 0.5 (표준) ⭐ 권장
```
장점:
  ✓ COCO 표준 기준
  ✓ 균형 잡힌 품질
  ✓ 대부분의 경우 적합
  ✓ 충분한 샘플 + 적절한 품질

단점:
  - 특별한 단점 없음

사용 시기:
  - 대부분의 경우 (기본값)
  - 표준 객체 탐지
  - 균형 잡힌 실험
```

### 0.6 (엄격)
```
장점:
  ✓ 고품질 Target만 선별
  ✓ 정확한 탐지만 학습
  ✓ 우수한 필터링 성능

단점:
  ✗ Target 샘플 부족 가능
  ✗ 학습 데이터 수집 어려움
  ✗ 너무 엄격한 기준

사용 시기:
  - YOLO 성능이 이미 좋은 경우
  - 고품질 필터링 원할 때
  - Target 샘플 충분히 수집 가능
```

### 0.7 (매우 엄격)
```
장점:
  ✓ 최고 품질만 선별
  ✓ 거의 완벽한 탐지만 Target

단점:
  ✗ Target 샘플 크게 부족
  ✗ 학습 실패 가능
  ✗ 너무 엄격

사용 시기:
  - YOLO 성능이 매우 좋은 경우만
  - 최고 품질 필요 시 (드물게)
  - 충분히 많은 이미지 (1000+)
```

---

## 🔍 임계값 선택 가이드

### Step 1: YOLO 초기 성능 확인

```bash
# Cycle 0 실행 (베이스라인)
python main.py  # skip_cycle_0=False
```

성능 확인:
```
Cycle 0: mAP50 = ?
```

### Step 2: 임계값 결정

| mAP50 | 추천 임계값 | 이유 |
|-------|-----------|------|
| < 0.3 | 0.3-0.4 | 성능 낮음, 관대한 기준 필요 |
| 0.3-0.5 | 0.4-0.5 | 보통 성능, 표준 기준 |
| 0.5-0.7 | **0.5** | 좋은 성능, 표준 권장 |
| > 0.7 | 0.5-0.6 | 우수 성능, 엄격 가능 |

### Step 3: 실험 후 조정

```python
# main.py 실행 후 로그 확인

Cycle 1: 멀티모달 학습 데이터 수집
수집 완료:
  - Target: ?개
  - Non-target: ?개
```

#### Target 부족 (< 50개)
```python
# 임계값 낮추기
multimodal_iou_threshold = 0.4  # 0.5 → 0.4
```

#### Non-target 부족 (< 50개)
```python
# 임계값 높이기
multimodal_iou_threshold = 0.6  # 0.5 → 0.6
```

#### 균형 잡힘 (80-120개씩)
```python
# 현재 값 유지
multimodal_iou_threshold = 0.5  # 유지
```

---

## 💡 실전 예시

### 예시 1: 표준 설정
```python
# 일반적인 경우
multimodal_iou_threshold = 0.5
multimodal_train_samples = 100

예상 결과:
  Target: 100개 (IoU 0.5~1.0)
  Non-target: 100개 (IoU 0.0~0.5)
```

### 예시 2: Target 부족 문제
```python
# 첫 실행
multimodal_iou_threshold = 0.5
multimodal_train_samples = 100

결과:
  Target: 35개 ← 부족!
  Non-target: 165개

# 해결: 임계값 낮추기
multimodal_iou_threshold = 0.4

재실행 결과:
  Target: 92개 ✓
  Non-target: 108개 ✓
```

### 예시 3: 고성능 YOLO
```python
# YOLO mAP50 = 0.75 (매우 좋음)

# 엄격한 기준 적용
multimodal_iou_threshold = 0.6

결과:
  Target: 98개 ✓ (고품질만)
  Non-target: 102개 ✓
```

---

## 📊 IoU 값 분포 이해

### 전형적인 분포 (잘 학습된 YOLO)

```
IoU 구간      | 탐지 수 | 분류 (기준 0.5)
-------------|---------|----------------
0.9 - 1.0    | 45개    | Target
0.8 - 0.9    | 35개    | Target
0.7 - 0.8    | 28개    | Target
0.6 - 0.7    | 22개    | Target
0.5 - 0.6    | 18개    | Target
-------------|---------|----------------
0.4 - 0.5    | 15개    | Non-target ←
0.3 - 0.4    | 12개    | Non-target
0.2 - 0.3    | 10개    | Non-target
0.1 - 0.2    | 8개     | Non-target
0.0 - 0.1    | 7개     | Non-target

Target 합계: 148개
Non-target 합계: 52개
```

### 저성능 YOLO의 분포

```
IoU 구간      | 탐지 수 | 분류 (기준 0.5)
-------------|---------|----------------
0.9 - 1.0    | 8개     | Target
0.8 - 0.9    | 12개    | Target
0.7 - 0.8    | 15개    | Target
0.6 - 0.7    | 18개    | Target
0.5 - 0.6    | 22개    | Target
-------------|---------|----------------
0.4 - 0.5    | 28개    | Non-target ←
0.3 - 0.4    | 35개    | Non-target
0.2 - 0.3    | 42개    | Non-target
0.1 - 0.2    | 50개    | Non-target
0.0 - 0.1    | 70개    | Non-target

Target 합계: 75개
Non-target 합계: 225개

→ 해결: 임계값 0.4로 낮추기
```

---

## 🛠️ 실전 워크플로우

### 1단계: 기본값으로 시작
```python
multimodal_iou_threshold = 0.5
```

### 2단계: Cycle 1 실행 후 로그 확인
```
수집 완료:
  - Target: ?개
  - Non-target: ?개
```

### 3단계: 필요시 조정

#### Target 너무 적음 (< 50)
```python
multimodal_iou_threshold = 0.4  # 낮추기
# 또는
multimodal_iou_threshold = 0.3  # 더 낮추기
```

#### Non-target 너무 적음 (< 50)
```python
multimodal_iou_threshold = 0.6  # 높이기
```

#### 균형 잡힘 (50-150개 범위)
```python
# 현재 값 유지
```

### 4단계: 재실행 및 성능 확인
```
Cycle 2: mAP50 = ?
→ 이전 사이클 대비 개선되었는가?
```

---

## 🎓 고급 팁

### 1. 데이터셋 크기 고려
```python
# 작은 데이터셋 (< 500 이미지)
multimodal_iou_threshold = 0.4  # 관대하게

# 보통 데이터셋 (500-2000)
multimodal_iou_threshold = 0.5  # 표준

# 큰 데이터셋 (> 2000)
multimodal_iou_threshold = 0.5-0.6  # 엄격 가능
```

### 2. 객체 크기 고려
```python
# 작은 객체 (차량이 이미지의 < 5%)
multimodal_iou_threshold = 0.4  # 관대하게

# 중간 크기 객체
multimodal_iou_threshold = 0.5  # 표준

# 큰 객체 (> 30%)
multimodal_iou_threshold = 0.6  # 엄격하게
```

### 3. 샘플 수 동시 조정
```python
# 데이터 부족 시
multimodal_iou_threshold = 0.4  # 낮춤
multimodal_train_samples = 50   # 줄임

# 데이터 충분 시
multimodal_iou_threshold = 0.5  # 표준
multimodal_train_samples = 200  # 늘림
```

---

## 📈 예상 성능 비교

| 임계값 | Target 품질 | Target 수량 | 필터링 성능 | 최종 mAP |
|-------|-----------|-----------|-----------|---------|
| 0.3 | 낮음 ⭐ | 많음 ⭐⭐⭐ | 낮음 ⭐ | +5% |
| 0.4 | 중하 ⭐⭐ | 많음 ⭐⭐⭐ | 중하 ⭐⭐ | +8% |
| **0.5** | **좋음 ⭐⭐⭐** | **충분 ⭐⭐** | **좋음 ⭐⭐⭐** | **+12%** |
| 0.6 | 우수 ⭐⭐⭐⭐ | 적음 ⭐ | 우수 ⭐⭐⭐⭐ | +10% |
| 0.7 | 최고 ⭐⭐⭐⭐⭐ | 매우 적음 | 최고 ⭐⭐⭐⭐⭐ | +6% |

**→ 0.5가 균형과 성능 모두 최고!**

---

## ✅ 체크리스트

설정 전:
- [ ] YOLO 초기 성능 확인 (Cycle 0)
- [ ] 데이터셋 크기 파악
- [ ] 객체 크기 고려

설정:
- [ ] main.py에서 `multimodal_iou_threshold` 설정
- [ ] 기본값 0.5로 시작 (권장)

실행 후:
- [ ] Target/Non-target 수량 확인
- [ ] 불균형 시 임계값 조정
- [ ] 성능 개선 확인

---

## 🎯 핵심 요약

```python
# main.py 설정

# ✅ 권장 (대부분의 경우)
multimodal_iou_threshold = 0.5

# Target 부족하면
multimodal_iou_threshold = 0.4

# Non-target 부족하면
multimodal_iou_threshold = 0.6
```

**기본값 0.5로 시작하고, 필요시 조정하세요!**

---

추가 질문이 있으면 알려주세요! 🚀
