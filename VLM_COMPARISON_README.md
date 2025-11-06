# VLM 모델 비교 스크립트 사용 가이드

이 스크립트는 여러 Vision-Language Model(VLM)의 이미지 캡션 생성 성능을 비교하기 위한 도구입니다.

## 지원 모델

| 모델 | 설명 | VRAM 요구사항 | 권장 GPU |
|------|------|---------------|----------|
| **BLIP** | 가볍고 빠른 기본 모델 | ~1GB | RTX 3060 이상 |
| **VIT-GPT2** | BLIP 대안, 비슷한 성능 | ~1GB | RTX 3060 이상 |
| **InstructBLIP** | 상세한 설명 지원, 7B 모델 | >16GB | RTX 3090/4090, A100 |
| **LLaVA** | 멀티모달 대화형 모델, 7B | >16GB | RTX 3090/4090, A100 |
| **Qwen-VL** | Qwen 기반 VLM, 상세 설명 | >16GB | RTX 3090/4090, A100 |

## 설치

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/auto_labeling

# 필요한 라이브러리 설치 (이미 설치되어 있을 수 있음)
pip install torch torchvision transformers pillow matplotlib opencv-python tqdm

# 대형 모델 사용 시 accelerate 필요
pip install accelerate
```

## 사용법

### 기본 사용법

```bash
# 1. 스크립트 상단의 CONFIG 섹션에서 설정 변경
# 2. 스크립트 실행
python compare_vlm_models.py
```

### CONFIG 설정 옵션

스크립트 파일(`compare_vlm_models.py`) 상단의 CONFIG 섹션에서 다음 변수들을 수정하세요:

```python
# 이미지 디렉토리 (샘플 이미지가 있는 폴더)
IMAGE_DIR = "./sample_images"

# 출력 디렉토리 (결과를 저장할 폴더)
OUTPUT_DIR = "./vlm_comparison_results"

# 테스트할 이미지 개수
NUM_IMAGES = 5

# 테스트할 모델 선택
# - None: 모든 모델 테스트 (VRAM >16GB 권장)
# - ['blip', 'vit-gpt2']: 경량 모델만 (RTX 3080 권장)
# - ['blip', 'instructblip']: 특정 모델만
MODELS_TO_TEST = ['blip', 'vit-gpt2']  # RTX 3080 기본 설정

# 사용할 GPU 번호 (0부터 시작)
GPU_NUM = 0
```

## 사용 예시

### 1. 경량 모델 테스트 (RTX 3080 기본 설정)

스크립트 상단 CONFIG 섹션:
```python
IMAGE_DIR = "./sample_images"
OUTPUT_DIR = "./vlm_comparison_results"
NUM_IMAGES = 5
MODELS_TO_TEST = ['blip', 'vit-gpt2']  # 기본값
GPU_NUM = 0
```

실행:
```bash
python compare_vlm_models.py
```

### 2. 모든 모델 테스트 (>16GB VRAM 필요)

스크립트 상단 CONFIG 섹션:
```python
IMAGE_DIR = "./sample_images"
OUTPUT_DIR = "./vlm_comparison_results"
NUM_IMAGES = 5
MODELS_TO_TEST = None  # 모든 모델
GPU_NUM = 0
```

실행:
```bash
python compare_vlm_models.py
```

### 3. 대형 모델만 테스트 (>16GB VRAM 필요)

스크립트 상단 CONFIG 섹션:
```python
IMAGE_DIR = "./sample_images"
OUTPUT_DIR = "./vlm_comparison_results"
NUM_IMAGES = 5
MODELS_TO_TEST = ['instructblip', 'llava', 'qwen-vl']
GPU_NUM = 0
```

실행:
```bash
python compare_vlm_models.py
```

### 4. 특정 모델만 테스트

스크립트 상단 CONFIG 섹션:
```python
IMAGE_DIR = "./sample_images"
OUTPUT_DIR = "./vlm_comparison_results"
NUM_IMAGES = 5
MODELS_TO_TEST = ['blip', 'instructblip']  # BLIP과 InstructBLIP만
GPU_NUM = 0
```

실행:
```bash
python compare_vlm_models.py
```

### 5. 이미지 개수 변경

스크립트 상단 CONFIG 섹션:
```python
IMAGE_DIR = "./sample_images"
OUTPUT_DIR = "./vlm_comparison_results"
NUM_IMAGES = 10  # 10개로 변경
MODELS_TO_TEST = ['blip', 'vit-gpt2']
GPU_NUM = 0
```

실행:
```bash
python compare_vlm_models.py
```

### 6. 다른 이미지 디렉토리 사용

스크립트 상단 CONFIG 섹션:
```python
IMAGE_DIR = "/path/to/my/images"  # 경로 변경
OUTPUT_DIR = "/path/to/output"
NUM_IMAGES = 5
MODELS_TO_TEST = ['blip', 'vit-gpt2']
GPU_NUM = 0
```

실행:
```bash
python compare_vlm_models.py
```

### 7. 특정 GPU 사용

스크립트 상단 CONFIG 섹션:
```python
IMAGE_DIR = "./sample_images"
OUTPUT_DIR = "./vlm_comparison_results"
NUM_IMAGES = 5
MODELS_TO_TEST = ['blip', 'vit-gpt2']
GPU_NUM = 1  # GPU 1번 사용
```

실행:
```bash
python compare_vlm_models.py
```

## 출력 결과

스크립트 실행 후 `output_dir`에 다음 파일들이 생성됩니다:

### 1. 개별 이미지 비교 결과
- `<이미지명>_comparison.png`: 각 이미지에 대한 모델별 캡션 비교

### 2. 전체 비교 테이블
- `comparison_table.html`: 모든 이미지와 모델의 캡션을 표 형식으로 보여주는 HTML 파일
  - 웹 브라우저에서 열어서 확인 가능
  - 이미지와 각 모델의 캡션을 한눈에 비교 가능

### 3. 통계 정보
- `statistics.txt`: 모델별 평균 캡션 길이, 단어 수 등의 통계 정보

## 예시 워크플로우

```bash
# 1. 샘플 이미지 디렉토리 준비
mkdir -p ~/vlm_test/samples
# 테스트할 이미지 5장을 ~/vlm_test/samples/ 에 복사

# 2. 스크립트의 CONFIG 섹션 수정 (경량 모델)
# compare_vlm_models.py 파일 열어서:
#   IMAGE_DIR = "~/vlm_test/samples"
#   OUTPUT_DIR = "~/vlm_test/results_light"
#   MODELS_TO_TEST = ['blip', 'vit-gpt2']

# 3. 경량 모델로 먼저 테스트 (빠름)
python compare_vlm_models.py

# 4. 결과 확인
firefox ~/vlm_test/results_light/comparison_table.html

# 5. VRAM이 충분하면 CONFIG 수정하여 대형 모델 테스트
# compare_vlm_models.py 파일 열어서:
#   OUTPUT_DIR = "~/vlm_test/results_large"
#   MODELS_TO_TEST = ['instructblip', 'llava']

# 6. 대형 모델 테스트
python compare_vlm_models.py

# 7. 결과 비교
firefox ~/vlm_test/results_large/comparison_table.html
```

## 주의사항

### GPU 메모리 부족 시

대형 모델(InstructBLIP, LLaVA, Qwen-VL)은 많은 VRAM을 요구합니다:

- **RTX 3080 (10GB)**: `blip`, `vit-gpt2`만 사용 권장
- **RTX 3090 (24GB)**: 모든 모델 사용 가능 (단, 한 번에 하나씩)
- **A100 (40GB/80GB)**: 모든 모델 동시 사용 가능

메모리 부족 오류가 발생하면:
1. CONFIG의 `MODELS_TO_TEST`를 경량 모델만 선택 (`['blip', 'vit-gpt2']`)
2. CONFIG의 `NUM_IMAGES`를 줄여서 테스트 (예: 3개)
3. 다른 GPU 프로세스 종료

### 모델 다운로드

처음 실행 시 Hugging Face에서 모델을 자동으로 다운로드합니다:
- BLIP: ~1GB
- VIT-GPT2: ~1GB
- InstructBLIP: ~14GB
- LLaVA: ~14GB
- Qwen-VL: ~14GB

충분한 디스크 공간과 안정적인 인터넷 연결이 필요합니다.

## 트러블슈팅

### ImportError 발생 시

```bash
# transformers 업데이트
pip install --upgrade transformers

# accelerate 설치
pip install accelerate
```

### CUDA Out of Memory 오류

스크립트 상단 CONFIG 섹션을 수정:

```python
# 경량 모델만 사용
MODELS_TO_TEST = ['blip', 'vit-gpt2']

# 또는 이미지 개수 줄이기
NUM_IMAGES = 3
```

그 후 실행:
```bash
python compare_vlm_models.py
```

### 특정 모델 로딩 실패

일부 모델이 로딩에 실패해도 나머지 모델로 계속 진행됩니다.
실패 메시지를 확인하고 필요한 라이브러리를 설치하세요.

## 결과 해석

### 캡션 품질 비교

- **BLIP / VIT-GPT2**: 간단하고 핵심적인 설명
  - 예: "a car parked on the street"

- **InstructBLIP / LLaVA / Qwen-VL**: 상세하고 구체적인 설명
  - 예: "a red sedan car parked on the side of a residential street, with trees in the background and a blue sky above"

### 사용 사례별 권장 모델

1. **빠른 처리가 필요한 경우**
   - BLIP 또는 VIT-GPT2

2. **상세한 설명이 필요한 경우**
   - InstructBLIP, LLaVA, Qwen-VL

3. **리소스 제약이 있는 경우**
   - BLIP (가장 경량)

4. **최고 품질이 필요한 경우**
   - InstructBLIP 또는 LLaVA (최신 모델)

## 라이선스 및 인용

각 모델의 라이선스를 확인하고 준수하세요:
- BLIP: BSD-3-Clause
- VIT-GPT2: Apache 2.0
- InstructBLIP: Salesforce Research License
- LLaVA: Apache 2.0
- Qwen-VL: Tongyi Qianwen License

## 참고 자료

- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [InstructBLIP Paper](https://arxiv.org/abs/2305.06500)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Qwen-VL Paper](https://arxiv.org/abs/2308.12966)

## 문의

이슈나 질문이 있으면 프로젝트 GitHub 저장소에 이슈를 등록해주세요.
