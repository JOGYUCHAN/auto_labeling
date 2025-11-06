"""
utils.py - 공통 유틸리티 함수 모듈
"""

import os
import random
import numpy as np
import torch
import time
import glob
import json
from typing import List, Tuple, Dict, Any

def set_seed(seed: int = 42):
    """전역 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device(gpu_num: int = 0) -> torch.device:
    """GPU 장치 반환"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")
    return torch.device("cpu")

def get_image_files(directory: str) -> List[str]:
    """이미지 파일 목록 반환"""
    if not os.path.exists(directory):
        return []
    
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    
    return sorted(files)

def get_model_files(directory: str, extension: str = ".pt") -> List[str]:
    """모델 파일 목록 반환"""
    if not os.path.exists(directory):
        return []
    return sorted(glob.glob(os.path.join(directory, f"*{extension}")))

def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """두 바운딩 박스의 IoU 계산"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def yolo_to_xyxy(yolo_box: Tuple, img_width: int, img_height: int) -> Tuple:
    """YOLO 형식을 xyxy 형식으로 변환"""
    cls_id, center_x, center_y, width, height = yolo_box
    x1 = int((center_x - width/2) * img_width)
    y1 = int((center_y - height/2) * img_height)
    x2 = int((center_x + width/2) * img_width)
    y2 = int((center_y + height/2) * img_height)
    return x1, y1, x2, y2

def xyxy_to_yolo(xyxy_box: Tuple, img_width: int, img_height: int) -> Tuple:
    """xyxy 형식을 YOLO 형식으로 변환"""
    x1, y1, x2, y2 = xyxy_box
    center_x = ((x1 + x2) / 2) / img_width
    center_y = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return center_x, center_y, width, height

def load_yolo_labels(label_path: str) -> List[Tuple]:
    """YOLO 라벨 파일 로드"""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 5:
                labels.append((
                    int(values[0]),
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                    float(values[4])
                ))
    return labels

def save_yolo_labels(labels: List[Tuple], label_path: str):
    """YOLO 라벨 파일 저장"""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for label in labels:
            line = ' '.join([str(x) for x in label])
            f.write(line + '\n')

def create_experiment_directory_structure(base_dir: str, max_cycles: int, 
                                         use_classifier: bool, 
                                         enable_classifier_retraining: bool):
    """실험 디렉토리 구조 생성"""
    directories = [base_dir]
    
    # 사이클별 디렉토리
    for cycle in range(max_cycles + 1):
        cycle_dir = os.path.join(base_dir, f"cycle_{cycle}")
        directories.extend([
            cycle_dir,
            os.path.join(cycle_dir, "detections"),
            os.path.join(cycle_dir, "labels")
        ])
        
        if use_classifier:
            directories.extend([
                os.path.join(cycle_dir, "filtered_detections"),
                os.path.join(cycle_dir, "filtered_labels")
            ])
            
            if enable_classifier_retraining:
                directories.extend([
                    os.path.join(cycle_dir, "cropped_objects", "class0"),
                    os.path.join(cycle_dir, "cropped_objects", "class1")
                ])
        
        if cycle > 0:
            directories.append(os.path.join(cycle_dir, "training"))
            if use_classifier and enable_classifier_retraining:
                directories.append(os.path.join(cycle_dir, "classification_training"))
            if use_classifier:  # 멀티모달 포함
                directories.append(os.path.join(cycle_dir, "multimodal_training"))
    
    # 데이터셋 디렉토리
    dataset_dir = os.path.join(base_dir, "dataset")
    directories.extend([
        os.path.join(dataset_dir, "images", "train"),
        os.path.join(dataset_dir, "images", "val"),
        os.path.join(dataset_dir, "labels", "train"),
        os.path.join(dataset_dir, "labels", "val")
    ])
    
    # 생성
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

class Timer:
    """실험 시간 측정 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.phase_times = {}
        self.current_phase = None
    
    def start(self):
        """타이머 시작"""
        self.start_time = time.time()
        return self
    
    def end(self):
        """타이머 종료"""
        self.end_time = time.time()
        if self.current_phase:
            self.end_phase()
        return self.get_total_time()
    
    def start_phase(self, phase_name: str):
        """단계 시작"""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_times[phase_name] = {'start': time.time()}
    
    def end_phase(self):
        """현재 단계 종료"""
        if self.current_phase and self.current_phase in self.phase_times:
            self.phase_times[self.current_phase]['end'] = time.time()
            self.phase_times[self.current_phase]['duration'] = (
                self.phase_times[self.current_phase]['end'] - 
                self.phase_times[self.current_phase]['start']
            )
        self.current_phase = None
    
    def get_total_time(self) -> float:
        """총 실행 시간 반환"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_summary(self) -> str:
        """시간 요약 반환"""
        total_time = self.get_total_time()
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        summary = [f"총 실행 시간: {hours}시간 {minutes}분 {seconds}초"]
        
        for phase_name, phase_data in self.phase_times.items():
            if 'duration' in phase_data:
                duration = phase_data['duration']
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                mins = int(duration // 60)
                secs = int(duration % 60)
                summary.append(f"  {phase_name}: {mins}분 {secs}초 ({percentage:.1f}%)")
        
        return "\n".join(summary)

def check_dependencies():
    """필수 라이브러리 확인"""
    required = ['torch', 'torchvision', 'ultralytics', 'cv2', 'PIL',
                'numpy', 'pandas', 'tqdm', 'yaml']

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise ImportError(f"필수 패키지 누락: {missing}")

    return True

def save_captions_to_json(captions_data: Dict[str, Any], save_path: str):
    """
    객체 캡션을 JSON 파일로 저장

    Args:
        captions_data: 캡션 데이터 딕셔너리
        save_path: 저장할 JSON 파일 경로
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, ensure_ascii=False, indent=2)

def load_captions_from_json(json_path: str) -> Dict[str, Any]:
    """
    JSON 파일에서 캡션 데이터 로드

    Args:
        json_path: JSON 파일 경로

    Returns:
        캡션 데이터 딕셔너리
    """
    if not os.path.exists(json_path):
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def append_caption_to_file(image_name: str, bbox: Tuple, caption: str,
                          class_id: int, confidence: float,
                          captions_file: str):
    """
    단일 객체 캡션을 JSON 파일에 추가

    Args:
        image_name: 이미지 파일명
        bbox: 바운딩 박스 (x1, y1, x2, y2)
        caption: 생성된 캡션
        class_id: 클래스 ID
        confidence: 신뢰도
        captions_file: 캡션 JSON 파일 경로
    """
    # 기존 데이터 로드
    captions_data = load_captions_from_json(captions_file)

    # 이미지 키가 없으면 생성
    if image_name not in captions_data:
        captions_data[image_name] = {"objects": []}

    # 객체 정보 추가
    obj_info = {
        "bbox": list(bbox),
        "caption": caption,
        "class": int(class_id),
        "confidence": float(confidence)
    }

    captions_data[image_name]["objects"].append(obj_info)

    # 저장
    save_captions_to_json(captions_data, captions_file)
