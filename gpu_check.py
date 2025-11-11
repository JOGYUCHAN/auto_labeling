"""
GPU 환경 확인 및 최적화 유틸리티
"""

import torch
import subprocess
import sys

def check_cuda_environment():
    """CUDA 환경 상세 체크"""
    print("="*60)
    print("CUDA Environment Check")
    print("="*60)
    
    # PyTorch CUDA 상태
    print(f"\n1. PyTorch CUDA 설정:")
    print(f"   - CUDA Available: {torch.cuda.is_available()}")
    print(f"   - PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   - CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"   - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   - GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"   - Name: {torch.cuda.get_device_name(i)}")
            print(f"   - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            # 메모리 사용량
            if torch.cuda.is_available():
                print(f"   - Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                print(f"   - Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    # NVIDIA Driver 버전
    print(f"\n2. NVIDIA Driver 정보:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # 상위 10줄만 출력
                if 'Driver Version' in line or 'CUDA Version' in line:
                    print(f"   {line.strip()}")
        else:
            print("   nvidia-smi 실행 실패")
    except FileNotFoundError:
        print("   nvidia-smi를 찾을 수 없습니다 (NVIDIA 드라이버 미설치)")
    
    # 권장사항
    print(f"\n3. 권장사항:")
    if not torch.cuda.is_available():
        print("   ⚠️  CUDA를 사용할 수 없습니다")
        print("   - CPU 모드로 실행됩니다 (느림)")
        print("   - 해결방법:")
        print("     1) NVIDIA 드라이버 업데이트 (권장)")
        print("     2) CUDA 11.8+ 설치")
        print("     3) PyTorch CUDA 버전 재설치")
        print("        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("   ✓ CUDA 사용 가능")
        
    print("="*60)

def get_optimal_batch_size(device_type='cpu'):
    """디바이스에 따른 최적 배치 크기 추천"""
    if device_type == 'cpu':
        return {
            'yolo_batch': 4,
            'classifier_batch': 8,
            'multimodal_batch': 4
        }
    else:
        # GPU 메모리 기반 추천
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 24:  # RTX 3090/4090
                return {
                    'yolo_batch': 32,
                    'classifier_batch': 64,
                    'multimodal_batch': 32
                }
            elif gpu_memory >= 12:  # RTX 3060 Ti
                return {
                    'yolo_batch': 16,
                    'classifier_batch': 32,
                    'multimodal_batch': 16
                }
            else:  # < 12GB
                return {
                    'yolo_batch': 8,
                    'classifier_batch': 16,
                    'multimodal_batch': 8
                }
        return {
            'yolo_batch': 16,
            'classifier_batch': 32,
            'multimodal_batch': 16
        }

def print_optimization_tips():
    """최적화 팁 출력"""
    print("\n" + "="*60)
    print("성능 최적화 팁")
    print("="*60)
    
    is_cuda = torch.cuda.is_available()
    
    print("\n1. 배치 크기 조정:")
    batch_sizes = get_optimal_batch_size('cuda' if is_cuda else 'cpu')
    print(f"   - YOLO: {batch_sizes['yolo_batch']}")
    print(f"   - Classifier: {batch_sizes['classifier_batch']}")
    print(f"   - Multimodal: {batch_sizes['multimodal_batch']}")
    
    print("\n2. 데이터 로딩:")
    print(f"   - num_workers: {2 if is_cuda else 0}")
    print("   - pin_memory: True (GPU 사용시)")
    
    print("\n3. 혼합 정밀도 학습 (Mixed Precision):")
    if is_cuda:
        print("   - torch.cuda.amp 사용 권장")
        print("   - 메모리 사용량 감소 + 속도 향상")
    else:
        print("   - CPU에서는 지원되지 않음")
    
    print("\n4. 모델 최적화:")
    print("   - YOLO: model.half() for FP16")
    print("   - Classifier: torch.compile() (PyTorch 2.0+)")
    
    print("="*60)

if __name__ == "__main__":
    check_cuda_environment()
    print_optimization_tips()