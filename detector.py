"""
detector.py - 객체 탐지 및 분류 통합 모듈 (간소화 버전)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict, Union
from classifier import ObjectClassifier
from captioning_classifier import ImageCaptioningClassifier
from multimodal_classifier import MultimodalFilterClassifier
from utils import xyxy_to_yolo

class ObjectDetector:
    """객체 탐지 및 분류 통합 클래스"""
    
    def __init__(self, model_path: str,
                 classifier: Optional[Union[ObjectClassifier, ImageCaptioningClassifier, MultimodalFilterClassifier]] = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.5):
        """탐지기 초기화"""
        self.model = YOLO(model_path)
        self.classifier = classifier
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.use_classifier = classifier is not None

        # 분류기 타입 확인
        self.classifier_type = "none"
        if isinstance(classifier, MultimodalFilterClassifier):
            self.classifier_type = "multimodal"
        elif isinstance(classifier, ImageCaptioningClassifier):
            self.classifier_type = "captioning"
        elif isinstance(classifier, ObjectClassifier):
            self.classifier_type = "traditional"
        
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'classification_calls': 0
        }
        
        print(f"객체 탐지기 초기화:")
        print(f"  - 분류기 타입: {self.classifier_type}")
    
    def detect_and_classify(self, image_path: str, cycle: int = 0):
        """객체 탐지 및 분류 수행"""
        img = cv2.imread(image_path)
        if img is None:
            return [], [], np.array([]), None
        
        h, w = img.shape[:2]
        
        # 객체 탐지
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False
        )
        
        detected_boxes = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detected_boxes.append({
                    'xyxy': [x1, y1, x2, y2],
                    'conf': conf,
                    'cls': 0
                })
        
        # 결과 처리
        detected_objects = []
        filtered_objects = []
        
        img_with_all_boxes = img.copy()
        img_with_filtered_boxes = img.copy() if self.use_classifier else None
        
        # Cycle 0에서는 필터링 비활성화
        apply_filtering = cycle > 0 and self.use_classifier
        
        for box_data in detected_boxes:
            x1, y1, x2, y2 = box_data['xyxy']
            conf = box_data['conf']
            cls_id = box_data['cls']
            
            obj_img = img[int(y1):int(y2), int(x1):int(x2)]
            center_x, center_y, width, height = xyxy_to_yolo((x1, y1, x2, y2), w, h)
            
            # 분류 필터링
            if apply_filtering and obj_img.size > 0:
                # 멀티모달 및 캡셔닝 분류기는 캡션도 반환
                if self.classifier_type in ["multimodal", "captioning"]:
                    image_name = os.path.basename(image_path)
                    bbox = (x1, y1, x2, y2)
                    pred_class, class_conf, caption = self.classifier.classify(obj_img, image_name, bbox, cycle)
                else:
                    pred_class, class_conf = self.classifier.classify(obj_img)

                self.stats['classification_calls'] += 1

                if pred_class == 0:  # 양성
                    detected_objects.append([cls_id, center_x, center_y, width, height])
                    self._draw_detection(img_with_all_boxes, x1, y1, x2, y2, conf, "Y", (0, 255, 0))
                    if img_with_filtered_boxes is not None:
                        if self.classifier_type == "traditional":
                            label = "C0"
                        elif self.classifier_type == "multimodal":
                            label = "M0"
                        else:
                            label = "K0"
                        self._draw_detection(img_with_filtered_boxes, x1, y1, x2, y2, class_conf, label, (0, 255, 0))
                else:  # 음성
                    filtered_objects.append([cls_id, center_x, center_y, width, height])
                    if self.classifier_type == "traditional":
                        label = "C1"
                    elif self.classifier_type == "multimodal":
                        label = "M1"
                    else:
                        label = "K1"
                    self._draw_detection(img_with_all_boxes, x1, y1, x2, y2, class_conf, label, (0, 0, 255))
                    self.stats['filtered_detections'] += 1
            else:
                detected_objects.append([cls_id, center_x, center_y, width, height])
                self._draw_detection(img_with_all_boxes, x1, y1, x2, y2, conf, "Y", (0, 255, 0))
        
        self.stats['total_detections'] += len(detected_boxes)
        
        return detected_objects, filtered_objects, img_with_all_boxes, img_with_filtered_boxes
    
    def _draw_detection(self, img: np.ndarray, x1: float, y1: float, 
                       x2: float, y2: float, conf: float, 
                       label: str, color: Tuple[int, int, int]):
        """탐지 결과 시각화"""
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{label}:{conf:.2f}",
                   (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def update_yolo_model(self, new_model_path: str):
        """YOLO 모델 업데이트"""
        self.model = YOLO(new_model_path)
    
    def update_classifier(self, new_classifier: Union[ObjectClassifier, ImageCaptioningClassifier, MultimodalFilterClassifier]):
        """분류 모델 업데이트"""
        self.classifier = new_classifier
        self.use_classifier = new_classifier is not None

        if isinstance(new_classifier, MultimodalFilterClassifier):
            self.classifier_type = "multimodal"
            print("분류기 업데이트: 멀티모달")
        elif isinstance(new_classifier, ImageCaptioningClassifier):
            self.classifier_type = "captioning"
            print("분류기 업데이트: 캡셔닝")
        elif isinstance(new_classifier, ObjectClassifier):
            self.classifier_type = "traditional"
            print("분류기 업데이트: 기존")
        else:
            self.classifier_type = "none"
    
    def get_stats(self) -> Dict:
        """탐지 통계 반환"""
        return self.stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_detections': 0,
            'filtered_detections': 0,
            'classification_calls': 0
        }

class CroppedObjectCollector:
    """크롭된 객체 수집 클래스"""
    
    def __init__(self, detector: ObjectDetector):
        self.detector = detector
    
    def collect_objects(self, image_files: List[str], output_dir: str, 
                       cycle: int, manual_label_dir: Optional[str] = None) -> Dict:
        """객체 크롭 이미지 수집"""
        import os
        from tqdm import tqdm
        
        class0_dir = os.path.join(output_dir, "class0")
        class1_dir = os.path.join(output_dir, "class1")
        os.makedirs(class0_dir, exist_ok=True)
        os.makedirs(class1_dir, exist_ok=True)
        
        collected = {'class0': 0, 'class1': 0}
        
        for image_file in tqdm(image_files, desc=f"객체 수집 (사이클 {cycle})"):
            img = cv2.imread(image_file)
            if img is None:
                continue
            
            # 객체 탐지
            results = self.detector.model.predict(
                source=img,
                conf=self.detector.conf_threshold,
                iou=self.detector.iou_threshold,
                save=False,
                verbose=False
            )
            
            detected_boxes = []
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detected_boxes.append((x1, y1, x2, y2, conf))
            
            # 객체 저장
            for idx, box_data in enumerate(detected_boxes):
                x1, y1, x2, y2, conf = box_data
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                h, w = img.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                obj_img = img[int(y1):int(y2), int(x1):int(x2)]
                
                if obj_img.size == 0 or obj_img.shape[0] < 10 or obj_img.shape[1] < 10:
                    continue
                
                # 분류
                if self.detector.classifier:
                    # 멀티모달 및 캡셔닝 분류기는 캡션도 반환
                    if self.detector.classifier_type in ["multimodal", "captioning"]:
                        image_name = os.path.basename(image_file)
                        bbox_coords = (x1, y1, x2, y2)
                        pred_class, class_conf, caption = self.detector.classifier.classify(obj_img, image_name, bbox_coords, cycle)
                    else:
                        pred_class, class_conf = self.detector.classifier.classify(obj_img)

                    if self.detector.classifier_type not in ["captioning", "multimodal"] and class_conf < 0.3:
                        continue
                else:
                    pred_class = 0
                    class_conf = conf
                
                # 파일명 생성
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                obj_filename = f"{base_name}_c{cycle}_o{idx}_cf{class_conf:.3f}.jpg"
                
                # 저장
                try:
                    if pred_class == 0:
                        save_path = os.path.join(class0_dir, obj_filename)
                        if cv2.imwrite(save_path, obj_img):
                            collected['class0'] += 1
                    else:
                        save_path = os.path.join(class1_dir, obj_filename)
                        if cv2.imwrite(save_path, obj_img):
                            collected['class1'] += 1
                except Exception:
                    continue
        
        print(f"\n객체 수집 완료: class0={collected['class0']}, class1={collected['class1']}")
        return collected
