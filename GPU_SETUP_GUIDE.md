# GPU 설정 가이드

## 현재 설정: GPU 2번 사용

main.py 파일 상단에서 GPU를 설정합니다.

## 기본 설정 (GPU 2번)

```python
# main.py 상단
import os

# GPU 설정 (가장 먼저 실행)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # GPU 2번 사용
```

이 설정으로:
- **물리적 GPU 2번**을 사용
- PyTorch에서는 **논리적으로 0번**으로 인식됨
- `gpu_num = 0`으로 설정

## GPU 변경 방법

### 다른 단일 GPU 사용

#### GPU 0번 사용
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_num = 0
```

#### GPU 1번 사용
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_num = 0
```

#### GPU 3번 사용
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpu_num = 0
```

### 여러 GPU 사용

#### GPU 2번과 3번 사용
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
gpu_num = 0  # 첫 번째 GPU (물리적 2번)
# gpu_num = 1  # 두 번째 GPU (물리적 3번)
```

#### GPU 0번, 1번, 2번 사용
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
gpu_num = 0  # 물리적 0번
# gpu_num = 1  # 물리적 1번
# gpu_num = 2  # 물리적 2번
```

## 설정 확인

실행 시 다음과 같은 정보가 출력됩니다:

```
============================================================
GPU 설정 정보
============================================================
CUDA_VISIBLE_DEVICES: 2
CUDA Available: True
Current CUDA Device: 0
Device Name: NVIDIA GeForce RTX 3090
Total GPU Memory: 24.00 GB
============================================================
```

## GPU 메모리 모니터링

### 실행 전 확인
```bash
nvidia-smi
```

### 실행 중 모니터링
```bash
watch -n 1 nvidia-smi
```

### 특정 GPU만 모니터링
```bash
watch -n 1 "nvidia-smi -i 2"
```

## 메모리 부족 시 해결책

### 1. 배치 크기 줄이기
```python
yolo_batch_size = 8          # 기본 16
multimodal_batch_size = 8    # 기본 16
classifier_batch_size = 8    # 기본 16
```

### 2. 학습 샘플 수 줄이기
```python
multimodal_train_samples = 50  # 기본 100
max_samples_per_class = 250    # 기본 500
```

### 3. 더 작은 YOLO 모델 사용
```python
# yolov8n.pt (가장 작음, 빠름)
# yolov8s.pt
# yolov8m.pt (중간)
# yolov8l.pt
# yolov8x.pt (가장 큼, 느림)
```

### 4. VLM 모델 변경
```python
# VIT-GPT2가 BLIP보다 메모리 효율적
multimodal_vlm_type = "vit-gpt2"  # BLIP보다 가벼움
```

## 코드 내 GPU 설정 위치

### main.py
```python
# 1. 환경 변수 (최상단)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 2. gpu_num 파라미터
gpu_num = 0  # 논리적 디바이스 번호
```

### config.py
```python
@dataclass
class ExperimentConfig:
    gpu_num: int = 0  # 기본값
```

### 런타임 확인
```python
import torch
print(f"Using GPU: {torch.cuda.current_device()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

## 멀티 GPU 학습 (고급)

현재 구현은 단일 GPU만 지원합니다. 멀티 GPU를 사용하려면:

### YOLO 멀티 GPU
```python
# active_learning.py의 train_yolo_model()에서
results = self.detector.model.train(
    data=yaml_path,
    epochs=self.config.yolo_epochs,
    device=[0, 1, 2, 3],  # 여러 GPU 사용
    # ...
)
```

### PyTorch DataParallel
```python
# multimodal_classifier.py에서
import torch.nn as nn

model = MultimodalAttentionClassifier(...)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
```

## 문제 해결

### "CUDA out of memory" 오류
```python
# 해결책 1: 배치 크기 감소
multimodal_batch_size = 4
yolo_batch_size = 4

# 해결책 2: GPU 2개 사용
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
```

### "CUDA device not available" 오류
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### GPU 점유 중
```bash
# 프로세스 확인
nvidia-smi

# 특정 프로세스 종료
kill -9 <PID>
```

## 성능 최적화

### GPU 사용률 확인
```bash
nvidia-smi dmon -i 2 -s u
```

### 최적 배치 크기 찾기
```python
# 메모리 사용량 90% 정도가 최적
# 너무 작으면: 느린 학습
# 너무 크면: OOM 오류

# 시작값
batch_size = 16

# OOM 발생 시
batch_size = 8, 4, 2, ...

# 여유 있으면
batch_size = 32, 64, ...
```

## 권장 설정

### GPU별 권장 배치 크기

| GPU | VRAM | YOLO Batch | Multimodal Batch |
|-----|------|------------|------------------|
| RTX 3060 | 12GB | 8 | 8 |
| RTX 3070 | 8GB | 4 | 4 |
| RTX 3080 | 10GB | 8 | 8 |
| RTX 3090 | 24GB | 16 | 16 |
| RTX 4090 | 24GB | 32 | 16 |
| A100 | 40GB | 32 | 32 |

### 현재 설정 (GPU 2번)

물리적 GPU 2번을 사용하도록 설정되어 있습니다:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpu_num = 0  # 논리적 번호
```

변경하려면 main.py 상단의 `CUDA_VISIBLE_DEVICES` 값을 수정하세요.

## 빠른 참조

### GPU 변경
```python
# main.py 4번째 줄
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 여기를 변경
```

### 확인 명령어
```bash
# GPU 상태
nvidia-smi

# 실험 실행
python main.py

# 첫 출력에서 GPU 정보 확인
# CUDA_VISIBLE_DEVICES: 2
# Device Name: ...
```

## 요약

✅ **현재 설정**: GPU 2번 사용  
✅ **변경 방법**: main.py 상단에서 `CUDA_VISIBLE_DEVICES` 수정  
✅ **확인 방법**: 실행 시 GPU 정보 출력  
✅ **메모리 부족**: 배치 크기 감소  

추가 질문이 있으면 알려주세요!
