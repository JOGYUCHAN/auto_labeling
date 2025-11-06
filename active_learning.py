"""
active_learning.py - 메인 Active Learning 클래스 (멀티모달 필터 통합)
"""

import os
import cv2
import yaml
import shutil
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
from ultralytics import YOLO
import time
import json

from config import ExperimentConfig
from utils import (set_seed, get_device, create_experiment_directory_structure,
                  get_image_files, save_yolo_labels, Timer, load_yolo_labels,
                  yolo_to_xyxy, calculate_iou)
from classifier import ObjectClassifier, ClassifierTrainer
from captioning_classifier import ImageCaptioningClassifier, CaptioningClassifierTrainer
from multimodal_classifier import MultimodalFilterClassifier, MultimodalFilterTrainer
from detector import ObjectDetector, CroppedObjectCollector
from evaluator import PerformanceEvaluator, MetricsManager

class YOLOActiveLearning:
    """YOLO Active Learning 메인 클래스 (멀티모달 필터 지원)"""
    
    def __init__(self, config: ExperimentConfig, model_path: str, 
                 classifier_path: Optional[str] = None):
        """Active Learning 시스템 초기화"""
        self.config = config
        self.model_path = model_path
        self.classifier_path = classifier_path
        
        set_seed(config.global_seed)
        self.device = get_device(config.gpu_num)
        
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.output_dir = os.path.join(config.output_dir, self.model_name)
        
        create_experiment_directory_structure(
            self.output_dir,
            config.max_cycles,
            config.use_classifier or config.use_captioning_classifier or config.use_multimodal_filter,
            config.enable_classifier_retraining and not config.use_captioning_classifier
        )
        
        self._initialize_components()
        
        self.timer = Timer()
        self.current_cycle = 0
        self.cycle_times = {}
        
        # 분류기 타입 결정
        classifier_type = "멀티모달" if config.use_multimodal_filter else \
                         "캡셔닝" if config.use_captioning_classifier else \
                         "기존" if config.use_classifier else "없음"
        
        print(f"Active Learning 초기화 완료")
        print(f"모델: {self.model_name}")
        print(f"분류기: {classifier_type}")
        print(f"출력: {self.output_dir}")
    
    def _initialize_components(self):
        """시스템 컴포넌트 초기화"""
        # 분류기 초기화
        if self.config.use_multimodal_filter:
            print("멀티모달 필터 초기화...")
            self.classifier = MultimodalFilterClassifier(
                vlm_model_type=self.config.multimodal_vlm_type,
                target_keywords=self.config.target_keywords,
                device=self.device,
                conf_threshold=self.config.class_conf_threshold,
                gpu_num=self.config.gpu_num
            )
            print("✓ 멀티모달 필터 초기화 완료")
            
        elif self.config.use_captioning_classifier:
            print("캡셔닝 분류기 초기화...")
            self.classifier = ImageCaptioningClassifier(
                target_keywords=self.config.target_keywords,
                model_type=self.config.captioning_model_type,
                device=self.device,
                conf_threshold=self.config.class_conf_threshold,
                gpu_num=self.config.gpu_num
            )
            print("✓ 캡셔닝 분류기 초기화 완료")
            
        elif self.config.use_classifier and self.classifier_path:
            print("기존 분류기 초기화...")
            self.classifier = ObjectClassifier(
                self.classifier_path,
                self.device,
                self.config.class_conf_threshold,
                self.config.gpu_num
            )
            print("✓ 기존 분류기 초기화 완료")
        else:
            self.classifier = None
            print("분류기 없음")
        
        # 객체 탐지기 초기화
        self.detector = ObjectDetector(
            self.model_path,
            self.classifier,
            self.config.conf_threshold,
            self.config.iou_threshold
        )
        
        # 분류 모델 트레이너 초기화
        if self.config.use_multimodal_filter:
            self.classifier_trainer = MultimodalFilterTrainer(
                self.device,
                self.config.multimodal_batch_size,
                self.config.multimodal_epochs,
                self.config.multimodal_learning_rate
            )
        elif self.config.use_captioning_classifier:
            self.classifier_trainer = CaptioningClassifierTrainer(self.device)
        elif self.config.use_classifier and self.config.enable_classifier_retraining:
            self.classifier_trainer = ClassifierTrainer(
                self.device,
                self.config.max_samples_per_class,
                self.config.classifier_batch_size,
                self.config.classifier_epochs,
                self.config.classifier_learning_rate_new,
                self.config.classifier_learning_rate_finetune
            )
        else:
            self.classifier_trainer = None
        
        # 크롭 객체 수집기
        if ((self.config.use_classifier and self.config.enable_classifier_retraining) or
            self.config.use_captioning_classifier or
            self.config.use_multimodal_filter):
            self.object_collector = CroppedObjectCollector(self.detector)
        else:
            self.object_collector = None
        
        # 성능 평가기
        self.evaluator = PerformanceEvaluator(
            self.detector,
            self.config.image_dir,
            self.config.label_dir,
            self.config.iou_threshold
        )
        
        # 메트릭 관리자
        self.metrics_manager = MetricsManager(self.output_dir)
        
        self.dataset_dir = os.path.join(self.output_dir, "dataset")
    
    def load_classifier_for_cycle(self, cycle: int) -> bool:
        """특정 사이클용 분류 모델 로드"""
        if not (self.config.use_classifier or self.config.use_captioning_classifier or 
                self.config.use_multimodal_filter):
            return False
        
        # 캡셔닝 분류기는 재학습하지 않음
        if self.config.use_captioning_classifier:
            print(f"Cycle {cycle}: 캡셔닝 분류기 사용")
            return True
        
        # 멀티모달 필터 로드
        if self.config.use_multimodal_filter:
            if cycle == 1:
                print(f"Cycle 1: 멀티모달 필터 학습 필요")
                return False
            
            cycle_model_path = os.path.join(
                self.output_dir, f"cycle_{cycle}", "multimodal_training", "multimodal_filter.pth"
            )
            
            if os.path.exists(cycle_model_path):
                print(f"Cycle {cycle}용 멀티모달 필터 로드: {cycle_model_path}")
                success = self.classifier.load_model(cycle_model_path)
                if success:
                    self.detector.update_classifier(self.classifier)
                return success
            else:
                print(f"⚠️ Cycle {cycle}용 멀티모달 필터 없음")
                return False
        
        # 기존 분류기
        if cycle == 1:
            print(f"Cycle 1: 초기 분류 모델 사용")
            return True
        
        cycle_model_path = os.path.join(
            self.output_dir, f"cycle_{cycle}", "classification_training", "best_classifier.pth"
        )
        
        if os.path.exists(cycle_model_path):
            print(f"Cycle {cycle}용 모델 로드: {cycle_model_path}")
            self.classifier = ObjectClassifier(
                cycle_model_path,
                self.device,
                self.config.class_conf_threshold,
                self.config.gpu_num
            )
            self.detector.update_classifier(self.classifier)
            return True
        else:
            print(f"⚠️ Cycle {cycle}용 모델 없음: {cycle_model_path}")
            return False
    
    def train_multimodal_filter(self, cycle: int) -> bool:
        """멀티모달 필터 학습"""
        if not self.config.use_multimodal_filter:
            return False
        
        if cycle != 1:
            print(f"⚠️ 멀티모달 필터는 Cycle 1에서만 학습")
            return False
        
        print(f"\n{'='*60}")
        print(f"Cycle {cycle}: 멀티모달 필터 학습 시작")
        print(f"{'='*60}")
        print(f"IoU 임계값: {self.config.multimodal_iou_threshold}")
        print(f"  - Target: IoU ≥ {self.config.multimodal_iou_threshold}")
        print(f"  - Non-target: IoU < {self.config.multimodal_iou_threshold}")
        
        # 1. 학습 데이터 수집 (IoU 기반)
        image_paths, labels = self.classifier_trainer.collect_training_data(
            self.config.image_dir,
            self.config.label_dir,
            self.detector,
            cycle,
            num_samples_per_class=self.config.multimodal_train_samples,
            iou_threshold=self.config.multimodal_iou_threshold  # 설정값 사용
        )
        
        if len(image_paths) < 20:
            print(f"✗ 학습 데이터 부족: {len(image_paths)}개")
            return False
        
        # 2. 모델 학습
        save_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "multimodal_training")
        success = self.classifier_trainer.train_model(
            self.classifier,
            image_paths,
            labels,
            save_dir
        )
        
        if success:
            # 학습된 모델 로드
            model_path = os.path.join(save_dir, "multimodal_filter.pth")
            self.classifier.load_model(model_path)
            self.detector.update_classifier(self.classifier)
            print(f"✓ Cycle {cycle}: 멀티모달 필터 학습 완료")
            return True
        
        return False
    
    # 나머지 메서드는 원본과 동일하게 유지...
    # (run_inference_cycle, _collect_objects_from_image, _calculate_metrics,
    #  train_classifier, prepare_yolo_dataset, train_yolo_model)
    
    def run_inference_cycle(self, cycle: int, collect_data: bool = False):
        """추론 사이클 실행"""
        cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle}")
        detections_dir = os.path.join(cycle_dir, "detections")
        labels_dir = os.path.join(cycle_dir, "labels")
        
        use_any_classifier = (self.config.use_classifier or 
                            self.config.use_captioning_classifier or 
                            self.config.use_multimodal_filter)
        
        if use_any_classifier:
            filtered_detections_dir = os.path.join(cycle_dir, "filtered_detections")
            filtered_labels_dir = os.path.join(cycle_dir, "filtered_labels")
        
        # 분류 데이터 수집 디렉토리
        collected_data = {'class0': 0, 'class1': 0}
        if collect_data and use_any_classifier:
            cropped_objects_dir = os.path.join(cycle_dir, "cropped_objects")
            class0_dir = os.path.join(cropped_objects_dir, "class0")
            class1_dir = os.path.join(cropped_objects_dir, "class1")
            os.makedirs(class0_dir, exist_ok=True)
            os.makedirs(class1_dir, exist_ok=True)
        
        # 이미지 파일 목록
        image_files = get_image_files(self.config.image_dir)
        if not image_files:
            print(f"이미지 없음: {self.config.image_dir}")
            return False, None
        
        detected_objects_count = 0
        filtered_objects_count = 0
        
        # 성능 평가용
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        # 진행률 표시
        desc_parts = [f"추론 (사이클 {cycle})"]
        if collect_data:
            desc_parts.append("+ 데이터 수집")
        if self.config.labels_available:
            desc_parts.append("+ 평가")
        desc = " ".join(desc_parts)
        
        with tqdm(total=len(image_files), desc=desc, unit="img") as pbar:
            for image_path in image_files:
                # 1. 객체 탐지 및 분류
                detected_objects, filtered_objects, img_all, img_filtered = \
                    self.detector.detect_and_classify(image_path, cycle)
                
                detected_objects_count += len(detected_objects)
                filtered_objects_count += len(filtered_objects)
                
                # 2. 결과 저장
                base_name = os.path.basename(image_path)
                if img_all is not None and img_all.size > 0:
                    cv2.imwrite(os.path.join(detections_dir, base_name), img_all)
                
                if use_any_classifier and img_filtered is not None and img_filtered.size > 0:
                    cv2.imwrite(os.path.join(filtered_detections_dir, base_name), img_filtered)
                
                # 3. 라벨 저장
                label_name = os.path.splitext(base_name)[0] + '.txt'
                
                if use_any_classifier:
                    all_objects = detected_objects + filtered_objects
                    save_yolo_labels(all_objects, os.path.join(labels_dir, label_name))
                    save_yolo_labels(detected_objects, os.path.join(filtered_labels_dir, label_name))
                else:
                    save_yolo_labels(detected_objects, os.path.join(labels_dir, label_name))
                
                # 4. 분류 데이터 수집 (멀티모달은 Cycle 1에서만)
                if collect_data and use_any_classifier:
                    if not self.config.use_multimodal_filter or cycle == 1:
                        img = cv2.imread(image_path)
                        if img is not None:
                            collected_count = self._collect_objects_from_image(
                                img, image_path, cycle, class0_dir, class1_dir
                            )
                            collected_data['class0'] += collected_count['class0']
                            collected_data['class1'] += collected_count['class1']
                
                # 5. 성능 평가
                if self.config.labels_available:
                    gt_label_path = os.path.join(self.config.label_dir, label_name)
                    if os.path.exists(gt_label_path):
                        gt_objects = load_yolo_labels(gt_label_path)
                        precision, recall, f1 = self._calculate_metrics(
                            detected_objects, gt_objects, image_path
                        )
                        all_precisions.append(precision)
                        all_recalls.append(recall)
                        all_f1_scores.append(f1)
                
                # 진행률 업데이트
                postfix = {'detected': detected_objects_count}
                if use_any_classifier:
                    postfix['filtered'] = filtered_objects_count
                if collect_data:
                    postfix['class0'] = collected_data['class0']
                    postfix['class1'] = collected_data['class1']
                
                pbar.set_postfix(postfix)
                pbar.update(1)
        
        # 성능 메트릭 계산
        if self.config.labels_available and all_precisions:
            avg_precision = np.mean(all_precisions)
            avg_recall = np.mean(all_recalls)
            avg_f1 = np.mean(all_f1_scores)
        else:
            avg_precision = -1.0
            avg_recall = -1.0
            avg_f1 = -1.0
        
        # 메트릭 저장
        metrics = {
            'Cycle': cycle,
            'Model': self.model_name,
            'mAP50': avg_precision,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1,
            'Detected_Objects': detected_objects_count,
            'Filtered_Objects': filtered_objects_count if cycle > 0 else 0,
            'Labels_Available': self.config.labels_available
        }
        
        self.metrics_manager.add_metrics(metrics)
        
        print(f"Cycle {cycle}: 탐지={detected_objects_count}, 필터링={filtered_objects_count}")
        if self.config.labels_available:
            print(f"성능: mAP50={avg_precision:.4f}")
        
        if collect_data:
            print(f"수집: class0={collected_data['class0']}, class1={collected_data['class1']}")
            return detected_objects_count > 0, collected_data
        
        return detected_objects_count > 0, None
    
    def _collect_objects_from_image(self, img: np.ndarray, image_path: str, 
                                   cycle: int, class0_dir: str, class1_dir: str) -> Dict:
        """단일 이미지에서 객체 수집"""
        collected = {'class0': 0, 'class1': 0}
        
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
                pred_class, class_conf = self.detector.classifier.classify(obj_img)
                if class_conf < 0.3:
                    continue
            else:
                pred_class = 0
                class_conf = conf
            
            # 저장
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            obj_filename = f"{base_name}_c{cycle}_o{idx}_cf{class_conf:.3f}.jpg"
            
            try:
                if pred_class == 0:
                    save_path = os.path.join(class0_dir, obj_filename)
                    if cv2.imwrite(save_path, obj_img):
                        collected['class0'] += 1
                else:
                    save_path = os.path.join(class1_dir, obj_filename)
                    if cv2.imwrite(save_path, obj_img):
                        collected['class1'] += 1
            except:
                continue
        
        return collected
    
    def _calculate_metrics(self, detected_objects: List, gt_objects: List, 
                          image_path: str) -> Tuple[float, float, float]:
        """개별 이미지 성능 지표 계산"""
        if len(gt_objects) == 0 and len(detected_objects) == 0:
            return 1.0, 1.0, 1.0
        elif len(gt_objects) == 0:
            return 0.0, 1.0, 0.0
        elif len(detected_objects) == 0:
            return 1.0, 0.0, 0.0
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # xyxy 변환
        gt_boxes = [yolo_to_xyxy(gt_obj, w, h) for gt_obj in gt_objects]
        pred_boxes = [yolo_to_xyxy(det_obj, w, h) for det_obj in detected_objects]
        
        # True Positive 계산
        tp = 0
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.config.iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
        
        precision = tp / len(pred_boxes)
        recall = tp / len(gt_boxes)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def train_classifier(self, target_cycle: int) -> bool:
        """특정 사이클용 분류 모델 학습"""
        if not ((self.config.use_classifier and self.config.enable_classifier_retraining) or
                self.config.use_captioning_classifier):
            return False
        
        data_cycle = target_cycle - 1
        cropped_data_dir = os.path.join(self.output_dir, f"cycle_{data_cycle}", "cropped_objects")
        
        # 캡셔닝 분류기
        if self.config.use_captioning_classifier:
            print(f"Target Cycle {target_cycle}용 캡셔닝 분류기 (재학습 없음)")
            
            new_classifier = self.classifier_trainer.train_classifier(
                cropped_data_dir,
                None,
                self.config.manual_label_dir,
                target_cycle,
                target_keywords=self.config.target_keywords,
                model_type=self.config.captioning_model_type
            )
            
            if new_classifier:
                classification_training_dir = os.path.join(
                    self.output_dir, f"cycle_{target_cycle}", "classification_training"
                )
                os.makedirs(classification_training_dir, exist_ok=True)
                config_save_path = os.path.join(classification_training_dir, "captioning_classifier_config.json")
                new_classifier.save_model(config_save_path)
                
                if target_cycle == self.current_cycle:
                    self.classifier = new_classifier
                    self.detector.update_classifier(new_classifier)
                
                print(f"✓ Cycle {target_cycle}용 캡셔닝 분류기 준비 완료")
                return True
            
            return False
        
        # 기존 분류기
        previous_model_path = None
        if target_cycle > 1:
            previous_model_path = os.path.join(
                self.output_dir, f"cycle_{target_cycle-1}", "classification_training", "best_classifier.pth"
            )
            if not os.path.exists(previous_model_path):
                previous_model_path = self.classifier_path
        else:
            previous_model_path = self.classifier_path
        
        print(f"Target Cycle {target_cycle}용 기존 분류 모델 학습")
        
        new_classifier = self.classifier_trainer.train_classifier(
            cropped_data_dir,
            previous_model_path,
            self.config.manual_label_dir,
            target_cycle
        )
        
        if new_classifier:
            classification_training_dir = os.path.join(
                self.output_dir, f"cycle_{target_cycle}", "classification_training"
            )
            os.makedirs(classification_training_dir, exist_ok=True)
            model_save_path = os.path.join(classification_training_dir, "best_classifier.pth")
            new_classifier.save_model(model_save_path)
            
            if target_cycle == self.current_cycle:
                self.classifier = new_classifier
                self.detector.update_classifier(new_classifier)
            
            print(f"✓ Cycle {target_cycle}용 기존 분류 모델 학습 완료")
            return True
        
        return False
    
    def prepare_yolo_dataset(self, cycle: int):
        """YOLO 학습용 데이터셋 준비"""
        use_any_classifier = (self.config.use_classifier or 
                            self.config.use_captioning_classifier or 
                            self.config.use_multimodal_filter)
        
        # 라벨 디렉토리 결정
        if use_any_classifier and cycle > 1:
            labels_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "filtered_labels")
        else:
            labels_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "labels")
        
        image_files = [os.path.basename(f) for f in get_image_files(self.config.image_dir)]
        
        # 학습/검증 분할
        if len(image_files) > 5:
            val_count = max(5, min(20, int(len(image_files) * 0.05)))
            val_files = image_files[:val_count]
            train_files = image_files[val_count:]
        else:
            val_files = image_files[:1]
            train_files = image_files[1:]
        
        # 데이터셋 복사
        for split, files in [("train", train_files), ("val", val_files)]:
            for file in files:
                # 이미지
                src_img = os.path.join(self.config.image_dir, file)
                dst_img = os.path.join(self.dataset_dir, "images", split, file)
                shutil.copy(src_img, dst_img)
                
                # 라벨
                label_name = os.path.splitext(file)[0] + '.txt'
                src_label = os.path.join(labels_dir, label_name)
                
                if os.path.exists(src_label):
                    dst_label = os.path.join(self.dataset_dir, "labels", split, label_name)
                    shutil.copy(src_label, dst_label)
        
        # YAML 파일
        dataset_yaml = {
            'path': os.path.abspath(self.dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['object']
        }
        
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"데이터셋 준비: 학습={len(train_files)}, 검증={len(val_files)}")
        return yaml_path
    
    def train_yolo_model(self, cycle: int) -> str:
        """YOLO 모델 학습"""
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        results_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "training")
        
        print(f"YOLO 모델 학습 시작 - Cycle {cycle}")
        
        results = self.detector.model.train(
            data=yaml_path,
            epochs=self.config.yolo_epochs,
            imgsz=640,
            batch=self.config.yolo_batch_size,
            patience=self.config.yolo_patience,
            project=results_dir,
            name="yolo_model",
            device=self.device
        )
        
        trained_model_path = os.path.join(results_dir, "yolo_model", "weights", "best.pt")
        self.detector.update_yolo_model(trained_model_path)
        
        print(f"YOLO 모델 학습 완료: {trained_model_path}")
        return trained_model_path
    
    def run(self, skip_cycle_0=False):
        """Active Learning 프로세스 실행"""
        print(f"\n{'='*60}")
        print(f"Active Learning 시작")
        print(f"모델: {self.model_name}")
        
        classifier_info = "멀티모달" if self.config.use_multimodal_filter else \
                         "캡셔닝" if self.config.use_captioning_classifier else \
                         "기존" if self.config.use_classifier else "없음"
        print(f"분류기: {classifier_info}")
        print(f"Cycle 0 건너뛰기: {skip_cycle_0}")
        print(f"{'='*60}")
        
        self.timer.start()
        
        try:
            start_cycle = 1 if skip_cycle_0 else 0
            
            if not skip_cycle_0:
                # Cycle 0: 베이스라인
                print(f"\n--- Cycle 0 - 베이스라인 ---")
                success, _ = self.run_inference_cycle(0, collect_data=False)
                if not success:
                    raise Exception("초기 추론 실패")
                print("Cycle 0 완료")
                start_cycle = 1
            else:
                print("\nCycle 0 건너뛰기")
            
            # 학습 사이클
            for cycle in range(start_cycle, self.config.max_cycles + 1):
                print(f"\n--- Cycle {cycle} ---")
                self.current_cycle = cycle
                
                cycle_inference_done = False
                
                # 0. Cycle 1 특별 처리
                if cycle == 1:
                    if skip_cycle_0:
                        print("Step: 초기 추론")
                        use_any_classifier = (self.config.use_classifier or 
                                            self.config.use_captioning_classifier or 
                                            self.config.use_multimodal_filter)
                        collect_data = use_any_classifier
                        success, collected_data = self.run_inference_cycle(1, collect_data=collect_data)
                        if not success:
                            continue
                        cycle_inference_done = True
                    
                    # 멀티모달 필터 학습 (Cycle 1)
                    if self.config.use_multimodal_filter:
                        print("\n멀티모달 필터 학습 (Cycle 1)")
                        self.train_multimodal_filter(cycle)
                
                # 1. 분류 모델 로드 (Cycle 2부터)
                elif cycle >= 2:
                    use_any_classifier = (self.config.use_classifier or 
                                        self.config.use_captioning_classifier or 
                                        self.config.use_multimodal_filter)
                    if use_any_classifier:
                        print(f"Step 1: Cycle {cycle}용 분류 모델 로드")
                        self.load_classifier_for_cycle(cycle)
                
                # 2. 통합 추론
                if not cycle_inference_done:
                    use_any_classifier = (self.config.use_classifier or 
                                        self.config.use_captioning_classifier or 
                                        self.config.use_multimodal_filter)
                    collect_data = (cycle >= 1 and use_any_classifier and 
                                  not self.config.use_multimodal_filter)  # 멀티모달은 한번만
                    
                    print(f"Step 2: 통합 추론")
                    success, collected_data = self.run_inference_cycle(cycle, collect_data=collect_data)
                    
                    if not success:
                        print(f"Cycle {cycle}: 탐지 없음")
                        continue
                
                # 3. 다음 사이클용 분류 모델 학습 (멀티모달 제외)
                if (cycle >= 1 and cycle < self.config.max_cycles):
                    if self.config.use_captioning_classifier:
                        print(f"Step 3: 캡셔닝 분류기는 재학습 없음")
                    elif self.config.use_multimodal_filter:
                        print(f"Step 3: 멀티모달 필터는 Cycle 1에서만 학습")
                    elif (self.config.use_classifier and
                          self.config.enable_classifier_retraining and
                          collected_data and
                          collected_data.get('class1', 0) >= 5):
                        print(f"Step 3: Cycle {cycle+1}용 기존 분류 모델 학습")
                        self.train_classifier(cycle + 1)
                
                # 4. YOLO 모델 학습
                print(f"Step 4: YOLO 모델 학습")
                self.prepare_yolo_dataset(cycle)
                self.train_yolo_model(cycle)
                
                print(f"Cycle {cycle} 완료")
            
            # 실험 완료
            total_time = self.timer.end()
            print(f"\n{'='*60}")
            print(f"Active Learning 완료!")
            print(f"총 소요 시간: {total_time/60:.1f}분")
            print(f"결과: {self.output_dir}")
            
            # 최종 성능
            best = self.metrics_manager.get_best_performance()
            if best:
                print(f"최고 성능: mAP50={best['mAP50']:.4f} (Cycle {best['Cycle']})")
            
            # 요약 파일
            summary_path = os.path.join(self.output_dir, "performance_summary.txt")
            self.metrics_manager.export_summary(summary_path)
            
            print(f"요약: {summary_path}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\nActive Learning 실행 중 오류: {str(e)}")
            raise
