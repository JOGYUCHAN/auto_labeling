#!/usr/bin/env python3
"""
compare_vlm_models.py - VLM 모델별 캡션 생성 결과 비교 스크립트

지원 모델:
- BLIP: 가볍고 빠름 (~1GB VRAM)
- VIT-GPT2: BLIP 대안, 비슷한 성능 (~1GB VRAM)
- InstructBLIP: 상세한 설명, 7B 모델 (>16GB VRAM 필요)
- LLaVA: 멀티모달 대화형 모델, 7B (>16GB VRAM 필요)
- Qwen-VL: Qwen 기반 VLM, 상세 설명 지원 (>16GB VRAM 필요)

사용법:
    python compare_vlm_models.py --image_dir ./samples --output_dir ./outputs --num_images 5

"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 모듈 임포트
try:
    from multimodal_classifier import MultimodalFilterClassifier
except ImportError:
    print("⚠️ multimodal_classifier 모듈을 찾을 수 없습니다.")
    print("   이 스크립트는 auto_labeling 프로젝트 디렉토리에서 실행해야 합니다.")
    sys.exit(1)


class VLMModelComparator:
    """VLM 모델 비교기"""

    # 지원 모델 목록 (VRAM 사용량 순서)
    MODELS = [
        {
            'name': 'blip',
            'display_name': 'BLIP',
            'description': '가볍고 빠름 (~1GB VRAM)',
            'vram': '~1GB'
        },
        {
            'name': 'vit-gpt2',
            'display_name': 'VIT-GPT2',
            'description': 'BLIP 대안, 비슷한 성능 (~1GB VRAM)',
            'vram': '~1GB'
        },
        {
            'name': 'instructblip',
            'display_name': 'InstructBLIP',
            'description': '상세한 설명, 7B 모델 (>16GB VRAM)',
            'vram': '>16GB'
        },
        {
            'name': 'llava',
            'display_name': 'LLaVA',
            'description': '멀티모달 대화형 모델 (>16GB VRAM)',
            'vram': '>16GB'
        },
        {
            'name': 'qwen-vl',
            'display_name': 'Qwen-VL',
            'description': 'Qwen 기반 VLM, 상세 설명 (>16GB VRAM)',
            'vram': '>16GB'
        }
    ]

    def __init__(self, device=None, models_to_test=None):
        """
        초기화

        Args:
            device: torch device (None이면 자동 선택)
            models_to_test: 테스트할 모델 리스트 (None이면 전체)
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # GPU 메모리 확인
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {gpu_mem_gb:.1f} GB")

            if gpu_mem_gb < 8:
                print("\n⚠️ 경고: GPU VRAM이 8GB 미만입니다.")
                print("   대형 모델(InstructBLIP, LLaVA, Qwen-VL)은 실행되지 않을 수 있습니다.")
                print("   --models blip vit-gpt2 옵션으로 경량 모델만 테스트하세요.\n")

        self.models_to_test = models_to_test if models_to_test else [m['name'] for m in self.MODELS]
        self.classifiers = {}

    def load_models(self):
        """선택된 모델들을 로드"""
        print(f"\n{'='*70}")
        print("VLM 모델 로딩 중...")
        print(f"{'='*70}\n")

        for model_info in self.MODELS:
            model_name = model_info['name']

            if model_name not in self.models_to_test:
                print(f"⊗ {model_info['display_name']}: 건너뜀")
                continue

            print(f"\n[{model_info['display_name']}] 로딩 중...")
            print(f"  - 설명: {model_info['description']}")
            print(f"  - VRAM: {model_info['vram']}")

            try:
                classifier = MultimodalFilterClassifier(
                    vlm_model_type=model_name,
                    device=self.device,
                    target_keywords=['object'],  # 일반적인 키워드
                    save_captions=False
                )

                self.classifiers[model_name] = classifier
                print(f"  ✓ {model_info['display_name']} 로딩 완료\n")

                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ✗ {model_info['display_name']} 로딩 실패: {e}")
                print(f"     이 모델은 테스트에서 제외됩니다.\n")
                continue

        if len(self.classifiers) == 0:
            print("\n⚠️ 로드된 모델이 없습니다. 종료합니다.")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"총 {len(self.classifiers)}개 모델 로드 완료")
        print(f"{'='*70}\n")

    def generate_captions_for_images(self, image_paths):
        """
        여러 이미지에 대해 각 모델의 캡션 생성

        Args:
            image_paths: 이미지 파일 경로 리스트

        Returns:
            results: {image_path: {model_name: caption}}
        """
        results = {}

        print(f"\n{'='*70}")
        print(f"{len(image_paths)}개 이미지에 대한 캡션 생성 중...")
        print(f"{'='*70}\n")

        for img_path in tqdm(image_paths, desc="이미지 처리"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 이미지 로드 실패: {img_path}")
                continue

            results[img_path] = {}

            for model_name, classifier in self.classifiers.items():
                try:
                    caption = classifier.generate_detailed_caption(img)
                    results[img_path][model_name] = caption

                    # 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️ 캡션 생성 실패 [{model_name}]: {img_path}")
                    print(f"   오류: {e}")
                    results[img_path][model_name] = f"[오류: {str(e)[:50]}]"

        return results

    def visualize_results(self, results, output_dir):
        """
        결과를 시각화하여 저장

        Args:
            results: {image_path: {model_name: caption}}
            output_dir: 출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)

        # 한글 폰트 설정 (matplotlib)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        print(f"\n{'='*70}")
        print("결과 시각화 중...")
        print(f"{'='*70}\n")

        # 1. 개별 이미지별 결과 저장
        for img_path, captions in tqdm(results.items(), desc="개별 결과 저장"):
            self._save_single_image_result(img_path, captions, output_dir)

        # 2. 전체 비교 테이블 생성
        self._save_comparison_table(results, output_dir)

        # 3. 통계 정보 저장
        self._save_statistics(results, output_dir)

        print(f"\n✓ 모든 결과가 저장되었습니다: {output_dir}")

    def _save_single_image_result(self, img_path, captions, output_dir):
        """개별 이미지의 모델별 캡션 결과 저장"""
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Figure 생성
        num_models = len(captions)
        fig_height = 3 + num_models * 1.5
        fig = plt.figure(figsize=(14, fig_height))

        # 이미지 표시
        ax_img = plt.subplot(num_models + 1, 1, 1)
        ax_img.imshow(img_rgb)
        ax_img.set_title(f"Image: {os.path.basename(img_path)}", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # 각 모델의 캡션 표시
        for idx, (model_name, caption) in enumerate(captions.items(), start=2):
            model_info = next((m for m in self.MODELS if m['name'] == model_name), None)
            display_name = model_info['display_name'] if model_info else model_name

            ax_text = plt.subplot(num_models + 1, 1, idx)
            ax_text.text(0.05, 0.5, f"{display_name}:\n{caption}",
                        fontsize=10, verticalalignment='center',
                        wrap=True, family='monospace')
            ax_text.axis('off')
            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(0, 1)

        plt.tight_layout()

        # 저장
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _save_comparison_table(self, results, output_dir):
        """전체 비교 테이블 저장 (HTML)"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VLM 모델 비교 결과</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e8f5e9;
        }
        .image-cell {
            text-align: center;
            vertical-align: middle;
        }
        .image-cell img {
            max-width: 200px;
            max-height: 150px;
            border-radius: 5px;
        }
        .caption-cell {
            font-size: 14px;
            line-height: 1.6;
        }
        .model-name {
            font-weight: bold;
            color: #2196F3;
        }
        .stats {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>VLM 모델 캡션 생성 비교</h1>
        <div class="stats">
            <strong>테스트 정보</strong><br>
            - 총 이미지 수: {num_images}<br>
            - 테스트 모델 수: {num_models}<br>
            - 테스트 모델: {model_list}
        </div>
"""

        # 모델 정보
        model_names = list(next(iter(results.values())).keys())
        model_list = ", ".join([next((m['display_name'] for m in self.MODELS if m['name'] == mn), mn)
                               for mn in model_names])

        html_content = html_content.format(
            num_images=len(results),
            num_models=len(model_names),
            model_list=model_list
        )

        # 테이블 생성
        html_content += """
        <table>
            <thead>
                <tr>
                    <th style="width: 15%;">이미지</th>
"""

        for model_name in model_names:
            model_info = next((m for m in self.MODELS if m['name'] == model_name), None)
            display_name = model_info['display_name'] if model_info else model_name
            html_content += f"                    <th>{display_name}</th>\n"

        html_content += """
                </tr>
            </thead>
            <tbody>
"""

        # 각 이미지 행 추가
        for img_path, captions in results.items():
            base_name = os.path.basename(img_path)
            comparison_img = os.path.splitext(base_name)[0] + "_comparison.png"

            html_content += f"""
                <tr>
                    <td class="image-cell">
                        <img src="{comparison_img}" alt="{base_name}"><br>
                        <small>{base_name}</small>
                    </td>
"""

            for model_name in model_names:
                caption = captions.get(model_name, "N/A")
                html_content += f"""
                    <td class="caption-cell">{caption}</td>
"""

            html_content += "                </tr>\n"

        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        # HTML 파일 저장
        html_path = os.path.join(output_dir, "comparison_table.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"  ✓ 비교 테이블 저장: {html_path}")

    def _save_statistics(self, results, output_dir):
        """통계 정보 저장"""
        stats_path = os.path.join(output_dir, "statistics.txt")

        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("VLM 모델 비교 통계\n")
            f.write("="*70 + "\n\n")

            f.write(f"총 이미지 수: {len(results)}\n")
            f.write(f"테스트 모델 수: {len(self.classifiers)}\n\n")

            f.write("="*70 + "\n")
            f.write("모델별 평균 캡션 길이\n")
            f.write("="*70 + "\n")

            # 모델별 캡션 길이 계산
            for model_name, classifier in self.classifiers.items():
                model_info = next((m for m in self.MODELS if m['name'] == model_name), None)
                display_name = model_info['display_name'] if model_info else model_name

                captions = [results[img_path][model_name]
                           for img_path in results
                           if model_name in results[img_path]]

                avg_length = sum(len(c) for c in captions) / len(captions) if captions else 0
                avg_words = sum(len(c.split()) for c in captions) / len(captions) if captions else 0

                f.write(f"\n{display_name}:\n")
                f.write(f"  - 평균 캡션 길이: {avg_length:.1f} 문자\n")
                f.write(f"  - 평균 단어 수: {avg_words:.1f} 단어\n")

            f.write("\n" + "="*70 + "\n")
            f.write("샘플 캡션\n")
            f.write("="*70 + "\n\n")

            # 첫 번째 이미지의 캡션 샘플 출력
            if results:
                first_img = next(iter(results.keys()))
                f.write(f"이미지: {os.path.basename(first_img)}\n\n")

                for model_name, caption in results[first_img].items():
                    model_info = next((m for m in self.MODELS if m['name'] == model_name), None)
                    display_name = model_info['display_name'] if model_info else model_name

                    f.write(f"[{display_name}]\n")
                    f.write(f"{caption}\n\n")

        print(f"  ✓ 통계 정보 저장: {stats_path}")


def get_image_files(image_dir, num_images=5):
    """이미지 파일 목록 가져오기"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(image_dir, file))

    if len(image_files) == 0:
        print(f"⚠️ {image_dir}에 이미지 파일이 없습니다.")
        return []

    # 최대 num_images개만 선택
    image_files = sorted(image_files)[:num_images]

    return image_files


def main():
    parser = argparse.ArgumentParser(
        description='VLM 모델별 이미지 캡션 생성 비교',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (모든 모델 테스트)
  python compare_vlm_models.py --image_dir ./samples --output_dir ./results

  # 경량 모델만 테스트 (RTX 3080 권장)
  python compare_vlm_models.py --image_dir ./samples --output_dir ./results --models blip vit-gpt2

  # 대형 모델만 테스트 (>16GB VRAM 필요)
  python compare_vlm_models.py --image_dir ./samples --output_dir ./results --models instructblip llava qwen-vl

  # 이미지 개수 지정
  python compare_vlm_models.py --image_dir ./samples --output_dir ./results --num_images 10
        """
    )

    parser.add_argument('--image_dir', type=str, required=True,
                       help='샘플 이미지가 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='결과를 저장할 디렉토리')
    parser.add_argument('--num_images', type=int, default=5,
                       help='테스트할 이미지 개수 (기본값: 5)')
    parser.add_argument('--models', nargs='+',
                       choices=['blip', 'vit-gpt2', 'instructblip', 'llava', 'qwen-vl'],
                       help='테스트할 모델 선택 (기본값: 전체)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='사용할 GPU 번호 (기본값: 0)')

    args = parser.parse_args()

    # 이미지 디렉토리 확인
    if not os.path.exists(args.image_dir):
        print(f"⚠️ 이미지 디렉토리를 찾을 수 없습니다: {args.image_dir}")
        return

    # 이미지 파일 가져오기
    image_files = get_image_files(args.image_dir, args.num_images)
    if len(image_files) == 0:
        print("⚠️ 처리할 이미지가 없습니다.")
        return

    print(f"\n{'='*70}")
    print("VLM 모델 비교 스크립트")
    print(f"{'='*70}")
    print(f"이미지 디렉토리: {args.image_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"테스트 이미지 수: {len(image_files)}")
    print(f"GPU: cuda:{args.gpu}" if torch.cuda.is_available() else "CPU 모드")
    print(f"{'='*70}\n")

    # Device 설정
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 비교기 생성
    comparator = VLMModelComparator(device=device, models_to_test=args.models)

    # 모델 로드
    comparator.load_models()

    # 캡션 생성
    results = comparator.generate_captions_for_images(image_files)

    # 결과 시각화
    comparator.visualize_results(results, args.output_dir)

    print(f"\n{'='*70}")
    print("완료!")
    print(f"{'='*70}")
    print(f"\n결과 확인:")
    print(f"  1. 개별 이미지 비교: {args.output_dir}/*_comparison.png")
    print(f"  2. 전체 비교 테이블: {args.output_dir}/comparison_table.html")
    print(f"  3. 통계 정보: {args.output_dir}/statistics.txt")
    print()


if __name__ == "__main__":
    main()
