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
python compare_vlm_models.py --image_dir <이미지_디렉토리> --output_dir <출력_디렉토리>
```

### 옵션

- `--image_dir`: 샘플 이미지가 있는 디렉토리 경로 (필수)
- `--output_dir`: 결과를 저장할 디렉토리 경로 (필수)
- `--num_images`: 테스트할 이미지 개수 (기본값: 5)
- `--models`: 테스트할 모델 선택 (기본값: 전체 모델)
  - 선택 가능: `blip`, `vit-gpt2`, `instructblip`, `llava`, `qwen-vl`
- `--gpu`: 사용할 GPU 번호 (기본값: 0)

## 사용 예시

### 1. 모든 모델 테스트 (>16GB VRAM 필요)

```bash
python compare_vlm_models.py \
    --image_dir ./sample_images \
    --output_dir ./vlm_comparison_results
```

### 2. 경량 모델만 테스트 (RTX 3080 권장)

```bash
python compare_vlm_models.py \
    --image_dir ./sample_images \
    --output_dir ./vlm_comparison_results \
    --models blip vit-gpt2
```

### 3. 대형 모델만 테스트 (>16GB VRAM 필요)

```bash
python compare_vlm_models.py \
    --image_dir ./sample_images \
    --output_dir ./vlm_comparison_results \
    --models instructblip llava qwen-vl
```

### 4. 특정 모델만 테스트

```bash
# BLIP과 InstructBLIP만 비교
python compare_vlm_models.py \
    --image_dir ./sample_images \
    --output_dir ./vlm_comparison_results \
    --models blip instructblip
```

### 5. 이미지 개수 지정

```bash
python compare_vlm_models.py \
    --image_dir ./sample_images \
    --output_dir ./vlm_comparison_results \
    --num_images 10
```

### 6. 특정 GPU 사용

```bash
python compare_vlm_models.py \
    --image_dir ./sample_images \
    --output_dir ./vlm_comparison_results \
    --gpu 1
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

# 2. 경량 모델로 먼저 테스트 (빠름)
python compare_vlm_models.py \
    --image_dir ~/vlm_test/samples \
    --output_dir ~/vlm_test/results_light \
    --models blip vit-gpt2

# 3. 결과 확인
firefox ~/vlm_test/results_light/comparison_table.html

# 4. VRAM이 충분하면 대형 모델도 테스트
python compare_vlm_models.py \
    --image_dir ~/vlm_test/samples \
    --output_dir ~/vlm_test/results_large \
    --models instructblip llava

# 5. 결과 비교
firefox ~/vlm_test/results_large/comparison_table.html
```

## 주의사항

### GPU 메모리 부족 시

대형 모델(InstructBLIP, LLaVA, Qwen-VL)은 많은 VRAM을 요구합니다:

- **RTX 3080 (10GB)**: `blip`, `vit-gpt2`만 사용 권장
- **RTX 3090 (24GB)**: 모든 모델 사용 가능 (단, 한 번에 하나씩)
- **A100 (40GB/80GB)**: 모든 모델 동시 사용 가능

메모리 부족 오류가 발생하면:
1. `--models` 옵션으로 경량 모델만 선택
2. `--num_images`를 줄여서 테스트
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

```bash
# 경량 모델만 사용
python compare_vlm_models.py \
    --image_dir ./samples \
    --output_dir ./results \
    --models blip vit-gpt2

# 또는 이미지 개수 줄이기
python compare_vlm_models.py \
    --image_dir ./samples \
    --output_dir ./results \
    --num_images 3
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
