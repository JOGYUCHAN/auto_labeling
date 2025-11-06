"""
config.py - Active Learning 실험 설정 관리 모듈 (멀티모달 필터 추가)
"""

import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ExperimentConfig:
    """실험 전체 설정 클래스"""
    
    # 경로 설정
    models_dir: str
    classifiers_dir: str
    image_dir: str
    label_dir: str
    output_dir: str
    manual_label_dir: Optional[str] = None
    
    # 하드웨어 설정
    gpu_num: int = 0
    
    # 학습 파라미터
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    class_conf_threshold: float = 0.5
    max_cycles: int = 10
    max_samples_per_class: int = 100
    
    # 분류기 설정
    use_classifier: bool = True
    enable_classifier_retraining: bool = False
    
    # 캡셔닝 분류기 설정
    use_captioning_classifier: bool = False
    captioning_model_type: str = "blip"  # "blip", "blip2", "instructblip", "vit-gpt2"
    target_keywords: List[str] = None
    
    # 멀티모달 필터 설정
    use_multimodal_filter: bool = False
    multimodal_vlm_type: str = "blip"  # "blip", "vit-gpt2", "instructblip", "llava", "qwen-vl"
    multimodal_train_samples: int = 100  # 클래스당 학습 샘플 수
    multimodal_iou_threshold: float = 0.5  # Target/Non-target 분류 IoU 임계값
    save_captions: bool = True  # VLM 캡션 저장 여부
    captions_output_dir: Optional[str] = None  # 캡션 저장 디렉토리 (None이면 output_dir/captions)
    
    # YOLO 학습 설정
    yolo_epochs: int = 50
    yolo_batch_size: int = 16
    yolo_patience: int = 10
    
    # 분류기 학습 설정 (기존 분류기만 해당)
    classifier_epochs: int = 15
    classifier_batch_size: int = 16
    classifier_learning_rate_new: float = 0.001
    classifier_learning_rate_finetune: float = 0.0001
    
    # 멀티모달 필터 학습 설정
    multimodal_epochs: int = 20
    multimodal_batch_size: int = 16
    multimodal_learning_rate: float = 0.001
    
    # 기타 설정
    global_seed: int = 42
    labels_available: bool = True
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.target_keywords is None:
            self.target_keywords = ['car']
    
    def validate(self):
        """설정 유효성 검사"""
        errors = []
        
        # 경로 존재 확인
        if not os.path.exists(self.models_dir):
            errors.append(f"YOLO 모델 디렉토리가 존재하지 않습니다: {self.models_dir}")
        
        if not os.path.exists(self.image_dir):
            errors.append(f"이미지 디렉토리가 존재하지 않습니다: {self.image_dir}")
        
        # 분류기 설정 검증
        active_classifiers = sum([
            self.use_classifier,
            self.use_captioning_classifier,
            self.use_multimodal_filter
        ])
        
        if active_classifiers > 1:
            errors.append("하나의 분류기만 활성화할 수 있습니다 (기본/캡셔닝/멀티모달 중 선택)")
        
        if self.use_classifier and not os.path.exists(self.classifiers_dir):
            errors.append(f"분류 모델 디렉토리가 존재하지 않습니다: {self.classifiers_dir}")
        
        if self.use_captioning_classifier:
            valid_models = ["blip", "blip2", "instructblip", "vit-gpt2"]
            if self.captioning_model_type not in valid_models:
                errors.append(f"캡셔닝 모델 타입은 {valid_models} 중 하나여야 합니다")
            
            if not self.target_keywords or len(self.target_keywords) == 0:
                errors.append("캡셔닝 분류기 사용 시 target_keywords가 필요합니다")
        
        if self.use_multimodal_filter:
            valid_vlm = ["blip", "vit-gpt2", "instructblip", "llava", "qwen-vl"]
            if self.multimodal_vlm_type not in valid_vlm:
                errors.append(f"멀티모달 VLM 타입은 {valid_vlm} 중 하나여야 합니다")
            
            if not self.target_keywords or len(self.target_keywords) == 0:
                errors.append("멀티모달 필터 사용 시 target_keywords가 필요합니다")
            
            if self.multimodal_train_samples < 10:
                errors.append("멀티모달 필터 학습 샘플은 최소 10개 이상이어야 합니다")
            
            if not 0.0 < self.multimodal_iou_threshold <= 1.0:
                errors.append("multimodal_iou_threshold는 0.0 초과 1.0 이하여야 합니다")
        
        # 파라미터 범위 확인
        if not 0.0 <= self.conf_threshold <= 1.0:
            errors.append("conf_threshold는 0.0과 1.0 사이여야 합니다")
        
        if not 0.0 <= self.iou_threshold <= 1.0:
            errors.append("iou_threshold는 0.0과 1.0 사이여야 합니다")
        
        if self.max_cycles <= 0:
            errors.append("max_cycles는 양수여야 합니다")
        
        if errors:
            raise ValueError("\n".join(errors))
    
    def get_summary(self):
        """설정 요약 반환"""
        summary = []
        summary.append("="*80)
        summary.append("실험 설정 요약")
        summary.append("="*80)
        summary.append(f"YOLO 모델: {self.models_dir}")
        summary.append(f"이미지: {self.image_dir}")
        summary.append(f"라벨: {self.label_dir} ({'사용' if self.labels_available else '미사용'})")
        summary.append(f"출력: {self.output_dir}")
        summary.append(f"최대 사이클: {self.max_cycles}")
        summary.append(f"GPU: {self.gpu_num}")
        summary.append("")
        
        # 분류기 설정
        summary.append("분류기 설정:")
        if self.use_multimodal_filter:
            summary.append(f"  - 타입: 멀티모달 필터 (VLM: {self.multimodal_vlm_type}, CNN: DenseNet121)")
            summary.append(f"  - 키워드: {', '.join(self.target_keywords)}")
            summary.append(f"  - 학습 샘플: 클래스당 {self.multimodal_train_samples}개")
            summary.append(f"  - IoU 임계값: {self.multimodal_iou_threshold} (Target: ≥{self.multimodal_iou_threshold}, Non-target: <{self.multimodal_iou_threshold})")
        elif self.use_captioning_classifier:
            summary.append(f"  - 타입: 캡셔닝 ({self.captioning_model_type})")
            summary.append(f"  - 키워드: {', '.join(self.target_keywords)}")
        elif self.use_classifier:
            summary.append(f"  - 타입: 기존 분류기")
            summary.append(f"  - 재학습: {self.enable_classifier_retraining}")
        else:
            summary.append(f"  - 타입: 사용 안함")
        
        summary.append("")
        summary.append(f"검출 신뢰도: {self.conf_threshold}")
        summary.append(f"IoU 임계값: {self.iou_threshold}")
        summary.append("="*80)
        
        return "\n".join(summary)
