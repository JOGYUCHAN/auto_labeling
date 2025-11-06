#!/bin/bash
# VLM 모델 비교 퀵스타트 스크립트

echo "========================================"
echo "VLM 모델 비교 퀵스타트"
echo "========================================"
echo ""

# 기본 설정
IMAGE_DIR=${1:-"./sample_images"}
OUTPUT_DIR=${2:-"./vlm_comparison_results"}
NUM_IMAGES=${3:-5}

# 이미지 디렉토리 확인
if [ ! -d "$IMAGE_DIR" ]; then
    echo "⚠️  이미지 디렉토리를 찾을 수 없습니다: $IMAGE_DIR"
    echo ""
    echo "사용법:"
    echo "  ./quickstart_vlm_comparison.sh <이미지_디렉토리> <출력_디렉토리> <이미지_개수>"
    echo ""
    echo "예시:"
    echo "  ./quickstart_vlm_comparison.sh ./my_images ./results 5"
    exit 1
fi

# 이미지 파일 개수 확인
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "⚠️  이미지 디렉토리에 이미지 파일이 없습니다: $IMAGE_DIR"
    exit 1
fi

echo "설정:"
echo "  - 이미지 디렉토리: $IMAGE_DIR"
echo "  - 출력 디렉토리: $OUTPUT_DIR"
echo "  - 발견된 이미지: $IMAGE_COUNT 개"
echo "  - 테스트할 이미지: $NUM_IMAGES 개"
echo ""

# GPU 메모리 확인
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""

    # VRAM 크기 확인 (MB 단위)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_GB=$((VRAM_MB / 1024))

    echo "감지된 VRAM: ${VRAM_GB}GB"
    echo ""

    if [ "$VRAM_GB" -lt 8 ]; then
        echo "⚠️  VRAM이 8GB 미만입니다. 경량 모델만 사용합니다."
        MODELS="blip vit-gpt2"
        echo "   선택된 모델: $MODELS"
    elif [ "$VRAM_GB" -lt 16 ]; then
        echo "ℹ️  VRAM이 8-16GB입니다. 경량 모델 사용을 권장합니다."
        echo "   1) 경량 모델만 (BLIP, VIT-GPT2)"
        echo "   2) 모든 모델 시도 (메모리 부족 가능)"
        echo ""
        read -p "   선택 (1/2, 기본값: 1): " choice

        if [ "$choice" = "2" ]; then
            MODELS=""
            echo "   모든 모델을 시도합니다. (메모리 부족 시 일부 실패 가능)"
        else
            MODELS="blip vit-gpt2"
            echo "   경량 모델만 사용합니다."
        fi
    else
        echo "✓ VRAM이 충분합니다. 모든 모델을 사용할 수 있습니다."
        MODELS=""
    fi
    echo ""
else
    echo "⚠️  nvidia-smi를 찾을 수 없습니다. CPU 모드로 실행됩니다."
    echo "   경량 모델만 사용합니다."
    MODELS="blip vit-gpt2"
    echo ""
fi

# Python 스크립트 실행
echo "========================================"
echo "VLM 모델 비교 시작..."
echo "========================================"
echo ""

if [ -n "$MODELS" ]; then
    # 특정 모델만 사용
    python compare_vlm_models.py \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_images "$NUM_IMAGES" \
        --models $MODELS
else
    # 모든 모델 사용
    python compare_vlm_models.py \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_images "$NUM_IMAGES"
fi

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "완료!"
    echo "========================================"
    echo ""
    echo "결과 파일:"
    echo "  📁 출력 디렉토리: $OUTPUT_DIR"
    echo "  📊 비교 테이블: $OUTPUT_DIR/comparison_table.html"
    echo "  📈 통계 정보: $OUTPUT_DIR/statistics.txt"
    echo ""
    echo "비교 테이블 보기:"
    echo "  firefox $OUTPUT_DIR/comparison_table.html"
    echo "  (또는 웹 브라우저로 comparison_table.html 파일 열기)"
    echo ""
else
    echo ""
    echo "⚠️  스크립트 실행 중 오류가 발생했습니다."
    echo "   자세한 내용은 위의 오류 메시지를 확인하세요."
    exit 1
fi
