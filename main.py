"""
main.py - 멀티모달 필터 지원 Active Learning 실험 실행기
"""

import os

# ==========================================
# CUDA 메모리 최적화 설정 (가장 먼저 실행)
# ==========================================
# CUDA 메모리 단편화 방지 및 효율적 할당 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# GPU 설정
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # GPU 2번 사용

import time
import traceback
from config import ExperimentConfig
from utils import check_dependencies, get_model_files, Timer
from active_learning import YOLOActiveLearning

class ExperimentRunner:
    """실험 실행기"""
    
    def __init__(self, check_deps=True):
        self.timer = Timer()
        
        if check_deps:
            try:
                check_dependencies()
                print("✓ 의존성 확인 완료")
            except ImportError as e:
                print(f"⚠️ 의존성 경고: {e}")
                print("⚠️ 일부 기능이 제한될 수 있습니다")
    
    def run_experiment(self, config: ExperimentConfig, skip_cycle_0=False):
        """단일 실험 실행"""
        print(f"\n{'='*80}")
        print("Active Learning 실험 시작")
        print(f"{'='*80}")
        print(config.get_summary())
        print(f"Cycle 0 건너뛰기: {skip_cycle_0}")
        
        # 설정 검증
        try:
            config.validate()
            print("✓ 설정 검증 완료")
        except ValueError as e:
            print(f"✗ 설정 오류: {e}")
            return False
        
        # 모델 파일 확인
        yolo_models = get_model_files(config.models_dir, ".pt")
        if not yolo_models:
            print(f"✗ YOLO 모델 없음: {config.models_dir}")
            return False
        
        # 분류기 확인
        classifier_models = []
        
        if config.use_multimodal_filter:
            print("✓ 멀티모달 필터 사용 - VLM 및 CNN 모델 자동 다운로드")
            print(f"  - VLM: {config.multimodal_vlm_type}")
            print(f"  - CNN: DenseNet121")
            print(f"  - 키워드: {config.target_keywords}")
            print(f"  - 학습 샘플: 클래스당 {config.multimodal_train_samples}개")
            classifier_models = [None]
            
        elif config.use_captioning_classifier:
            print("✓ 캡셔닝 분류기 사용 - 사전훈련 모델 자동 다운로드")
            classifier_models = [None]
            
        elif config.use_classifier:
            classifier_models = get_model_files(config.classifiers_dir, ".pth")
            if not classifier_models:
                print(f"✗ 분류 모델 없음: {config.classifiers_dir}")
                return False
            
            for clf_path in classifier_models:
                if not os.path.exists(clf_path):
                    print(f"✗ 분류 모델 파일 없음: {clf_path}")
                    return False
                else:
                    print(f"✓ 분류 모델 확인: {os.path.basename(clf_path)}")
        else:
            classifier_models = [None]
        
        print(f"✓ YOLO 모델 {len(yolo_models)}개")
        
        if config.use_multimodal_filter:
            print(f"✓ 멀티모달 필터: VLM={config.multimodal_vlm_type}, CNN=DenseNet121")
        elif config.use_captioning_classifier:
            print(f"✓ 캡셔닝 분류기: {config.captioning_model_type}, 키워드: {config.target_keywords}")
        elif config.use_classifier:
            print(f"✓ 분류 모델 {len(classifier_models)}개")
        
        # 실험 실행
        self.timer.start()
        success_count = 0
        total_count = 0
        
        for classifier_path in classifier_models:
            # 분류기 이름
            if config.use_multimodal_filter:
                classifier_name = f"multimodal_{config.multimodal_vlm_type}"
            elif config.use_captioning_classifier:
                classifier_name = f"captioning_{config.captioning_model_type}"
            elif classifier_path is None:
                classifier_name = "no_classifier"
            else:
                classifier_name = os.path.splitext(os.path.basename(classifier_path))[0]
            
            for model_path in yolo_models:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                total_count += 1
                
                cycle_info = "Cycle 1부터" if skip_cycle_0 else "Cycle 0부터"
                print(f"\n--- 실험 {total_count}: {model_name} + {classifier_name} ({cycle_info}) ---")
                
                try:
                    # 개별 실험 설정
                    experiment_config = ExperimentConfig(**config.__dict__)
                    experiment_config.output_dir = os.path.join(
                        config.output_dir,
                        f"{classifier_name}_{model_name}"
                    )
                    
                    # Active Learning 실행
                    al = YOLOActiveLearning(experiment_config, model_path, classifier_path)
                    al.run(skip_cycle_0=skip_cycle_0)
                    
                    success_count += 1
                    print(f"✓ 실험 완료: {model_name}")
                    
                except Exception as e:
                    print(f"✗ 실험 실패: {str(e)}")
                    
                    # 오류 로그
                    error_dir = os.path.join(
                        config.output_dir,
                        f"{classifier_name}_{model_name}",
                        "error_logs"
                    )
                    os.makedirs(error_dir, exist_ok=True)
                    
                    with open(os.path.join(error_dir, "error.log"), "w") as f:
                        f.write(f"오류 발생: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"오류: {str(e)}\n\n")
                        f.write(f"상세:\n{traceback.format_exc()}")
        
        # 실험 완료
        total_time = self.timer.end()
        
        print(f"\n{'='*80}")
        print("실험 완료!")
        print(f"성공: {success_count}/{total_count}")
        print(f"총 소요 시간: {total_time/60:.1f}분")
        print(f"결과: {config.output_dir}")
        print(f"{'='*80}")
        
        return success_count > 0

def main():
    """메인 실험 실행 함수"""
    
    # GPU 정보 출력
    import torch
    print(f"\n{'='*60}")
    print("GPU 설정 정보")
    print(f"{'='*60}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")  # 0은 가시적인 첫 번째 디바이스
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*60}\n")
    
    # ==========================================
    # 실험 파라미터 설정
    # ==========================================
    
    models_dir = "../model weights/YOLO/coco_vehicle"
    classifiers_dir = "../model weights/Classification/visdrone"
    image_dir = "../data/car_object_detection/images"
    label_dir = "../data/car_object_detection/labels"
    output_dir = "../experiment_results/kaggle_multi_modal"
    
    # 기본 파라미터
    conf_threshold = 0.25
    iou_threshold = 0.5
    class_conf_threshold = 0.5
    max_cycles = 10
    gpu_num = 2 
    
    # ==========================================
    # 분류기 설정 (3가지 중 1개만 선택)
    # ==========================================
    
    # 1. 기존 분류기
    use_classifier = False
    enable_classifier_retraining = False
    
    # 2. 캡셔닝 분류기
    use_captioning_classifier = False
    captioning_model_type = "vit-gpt2"  # "blip" 또는 "vit-gpt2"
    
    # 3. 멀티모달 필터 (NEW!)
    use_multimodal_filter = True

    # VLM 모델 선택 (아래 중 하나 선택)
    multimodal_vlm_type = "qwen-vl"  # 추천 모델 옵션:
    # - "blip": 가볍고 빠름 (기본)
    # - "vit-gpt2": BLIP 대안, 비슷한 성능
    # - "instructblip": 상세한 설명, 7B 모델 (VRAM 요구량 높음)
    # - "llava": 멀티모달 대화형 모델, 7B (VRAM 요구량 높음)
    # - "qwen-vl": Qwen 기반 VLM, 상세 설명 지원 ✓ 추천!

    multimodal_train_samples = 100  # 클래스당 IoU 기반 학습 샘플 수

    # 멀티모달 Target/Non-target 분류 IoU 임계값
    multimodal_iou_threshold = 0.5  # Target: ≥0.5, Non-target: <0.5
    # 권장값:
    # - 0.3: 매우 관대 (더 많은 Target 샘플)
    # - 0.4: 관대 (충분한 Target 샘플)
    # - 0.5: 표준 (COCO 기준, 권장) ✓
    # - 0.6: 엄격 (고품질 Target만)
    # - 0.7: 매우 엄격 (최고 품질만, Target 부족 가능)

    # 캡션 저장 설정
    save_captions = True  # VLM이 생성한 객체 설명을 JSON 파일로 저장
    captions_output_dir = None  # None이면 output_dir/captions 사용

    # 공통 설정
    target_keywords = ["car", "vehicle"]  # 양성 객체 키워드
    
    # 기타 설정
    skip_cycle_0 = False  # Cycle 0 건너뛰기
    
    yolo_epochs = 50
    yolo_batch_size = 16
    yolo_patience = 10
    
    classifier_epochs = 20
    classifier_batch_size = 16
    max_samples_per_class = 500
    
    # 멀티모달 학습 파라미터
    multimodal_epochs = 20
    multimodal_batch_size = 16
    multimodal_learning_rate = 0.001
    
    global_seed = 42
    
    # ==========================================
    # 설정 검증
    # ==========================================
    
    active_classifiers = sum([use_classifier, use_captioning_classifier, use_multimodal_filter])
    if active_classifiers > 1:
        print("✗ 오류: 하나의 분류기만 활성화할 수 있습니다")
        print("  - use_classifier")
        print("  - use_captioning_classifier")
        print("  - use_multimodal_filter")
        return
    
    if use_multimodal_filter:
        valid_vlm = ["blip", "vit-gpt2", "instructblip", "llava", "qwen-vl"]
        if multimodal_vlm_type not in valid_vlm:
            print(f"✗ 오류: 멀티모달 VLM은 {valid_vlm} 중 하나여야 합니다")
            return

        if not target_keywords:
            print("✗ 오류: 멀티모달 필터 사용 시 target_keywords 필요")
            return

        print(f"\n✓ 멀티모달 필터 설정:")
        print(f"   - VLM 모델: {multimodal_vlm_type}")
        print(f"   - CNN 모델: DenseNet121")
        print(f"   - 키워드: {target_keywords}")
        print(f"   - 학습 샘플: 클래스당 {multimodal_train_samples}개 (IoU 기반)")
        print(f"   - IoU 임계값: {multimodal_iou_threshold}")
        print(f"     · Target: IoU ≥ {multimodal_iou_threshold}")
        print(f"     · Non-target: IoU < {multimodal_iou_threshold}")
        print(f"   - 캡션 저장: {'활성화' if save_captions else '비활성화'}")
        if save_captions:
            caption_dir = captions_output_dir if captions_output_dir else os.path.join(output_dir, "captions")
            print(f"   - 캡션 저장 위치: {caption_dir}")
        print(f"   - Cycle 1에서 GT와 IoU 비교하여 자동 분류")
    
    elif use_captioning_classifier:
        valid_models = ["blip", "vit-gpt2"]
        if captioning_model_type not in valid_models:
            print(f"✗ 오류: 캡셔닝 모델은 {valid_models} 중 하나여야 합니다")
            return
        
        if not target_keywords:
            print("✗ 오류: 캡셔닝 분류기 사용 시 target_keywords 필요")
            return
        
        print(f"✓ 캡셔닝 분류기 설정:")
        print(f"   - 모델: {captioning_model_type}")
        print(f"   - 키워드: {target_keywords}")
    
    # ==========================================
    # 라벨 확인
    # ==========================================
    
    labels_available = False
    if os.path.exists(label_dir):
        try:
            label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            labels_available = len(label_files) > 0
        except:
            labels_available = False
    
    if not labels_available:
        print(f"\n⚠️ 라벨 없음: {label_dir}")
        if use_multimodal_filter:
            print("✗ 멀티모달 필터는 Cycle 1에서 IoU 기반 학습을 위해 라벨이 필수입니다")
            return
        print("⚠️ 성능 평가 없이 진행됩니다")
        
        response = input("\n계속하시겠습니까? (y/n): ").lower().strip()
        if response != 'y':
            print("취소됨")
            return
    else:
        label_count = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        print(f"✓ 라벨: {label_count}개")
        
        if use_multimodal_filter:
            print(f"✓ 멀티모달 필터: Cycle 1에서 IoU 기반 학습 데이터 생성 가능")
    
    # ==========================================
    # 실험 설정 생성
    # ==========================================
    
    config = ExperimentConfig(
        models_dir=models_dir,
        classifiers_dir=classifiers_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        labels_available=labels_available,
        gpu_num=gpu_num,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        class_conf_threshold=class_conf_threshold,
        max_cycles=max_cycles,
        max_samples_per_class=max_samples_per_class,
        use_classifier=use_classifier,
        enable_classifier_retraining=enable_classifier_retraining,
        use_captioning_classifier=use_captioning_classifier,
        captioning_model_type=captioning_model_type,
        use_multimodal_filter=use_multimodal_filter,
        multimodal_vlm_type=multimodal_vlm_type,
        multimodal_train_samples=multimodal_train_samples,
        multimodal_iou_threshold=multimodal_iou_threshold,
        save_captions=save_captions,
        captions_output_dir=captions_output_dir,
        target_keywords=target_keywords,
        yolo_epochs=yolo_epochs,
        yolo_batch_size=yolo_batch_size,
        yolo_patience=yolo_patience,
        classifier_epochs=classifier_epochs,
        classifier_batch_size=classifier_batch_size,
        multimodal_epochs=multimodal_epochs,
        multimodal_batch_size=multimodal_batch_size,
        multimodal_learning_rate=multimodal_learning_rate,
        global_seed=global_seed
    )
    
    # ==========================================
    # 실험 실행
    # ==========================================
    
    print("\nYOLO Active Learning with Multimodal Filter")
    print("="*60)
    print(f"Cycle 0 건너뛰기: {skip_cycle_0}")
    
    if use_multimodal_filter:
        print(f"🧠 멀티모달 필터:")
        print(f"   - VLM: {multimodal_vlm_type}")
        print(f"   - CNN: DenseNet121")
        print(f"   - 키워드: {target_keywords}")
        print(f"   - IoU 임계값: {multimodal_iou_threshold}")
        print(f"   - 학습: Cycle 1에서 IoU 기반 샘플링")
    elif use_captioning_classifier:
        print(f"🔤 캡셔닝 분류기: {captioning_model_type}")
        print(f"🎯 키워드: {target_keywords}")
    elif use_classifier:
        print(f"🧠 기존 분류기: 재학습 {'활성화' if enable_classifier_retraining else '비활성화'}")
    else:
        print("🚫 분류기 없음")
    
    if skip_cycle_0:
        print("⚡ 빠른 모드: Cycle 1부터 시작")
    else:
        print("📊 표준 모드: Cycle 0 베이스라인 포함")
    
    try:
        runner = ExperimentRunner(check_deps=True)
        success = runner.run_experiment(config, skip_cycle_0=skip_cycle_0)
        
        if success:
            print("\n🎉 실험 성공!")
        else:
            print("\n❌ 실험 실패")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {str(e)}")
        print("\n상세:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
