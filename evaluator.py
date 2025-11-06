"""
evaluator.py - 성능 평가 모듈 (간소화 버전)
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
from detector import ObjectDetector
from utils import load_yolo_labels, yolo_to_xyxy, calculate_iou

class PerformanceEvaluator:
    """성능 평가 클래스"""
    
    def __init__(self, detector: ObjectDetector, image_dir: str, 
                 label_dir: str, iou_threshold: float = 0.5):
        """성능 평가기 초기화"""
        self.detector = detector
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.iou_threshold = iou_threshold
        
        self.labels_available = (
            os.path.exists(label_dir) and
            len([f for f in os.listdir(label_dir) if f.endswith('.txt')]) > 0
        )
    
    def evaluate(self, cycle: int, model_name: str) -> Dict:
        """성능 평가 수행"""
        if not self.labels_available:
            print(f"⚠️ 라벨 없음 - Cycle {cycle} 성능 평가 생략")
            return {
                'Cycle': cycle,
                'Model': model_name,
                'mAP50': -1.0,
                'Precision': -1.0,
                'Recall': -1.0,
                'F1-Score': -1.0,
                'Detected_Objects': 0,
                'Filtered_Objects': 0,
                'Labels_Available': False
            }
        
        image_files = [f for f in os.listdir(self.image_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        total_detected = 0
        total_filtered = 0
        
        original_stats = self.detector.get_stats()
        self.detector.reset_stats()
        
        for image_file in tqdm(image_files, desc=f"평가 (사이클 {cycle})", leave=False):
            image_path = os.path.join(self.image_dir, image_file)
            
            detected_objects, filtered_objects, _, _ = self.detector.detect_and_classify(image_path, cycle)
            
            total_detected += len(detected_objects)
            total_filtered += len(filtered_objects)
            
            # 정답 라벨
            gt_label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + '.txt')
            if os.path.exists(gt_label_path):
                gt_objects = load_yolo_labels(gt_label_path)
                precision, recall, f1 = self._calculate_metrics(detected_objects, gt_objects, image_path)
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1_scores.append(f1)
            else:
                all_precisions.append(0.0)
                all_recalls.append(0.0)
                all_f1_scores.append(0.0)
        
        self.detector.stats = original_stats
        
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1_scores)
        
        filtered_count = total_filtered if cycle > 0 else 0
        
        return {
            'Cycle': cycle,
            'Model': model_name,
            'mAP50': avg_precision,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1,
            'Detected_Objects': total_detected,
            'Filtered_Objects': filtered_count,
            'Labels_Available': True
        }
    
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
            
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
        
        precision = tp / len(pred_boxes)
        recall = tp / len(gt_boxes)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

class MetricsManager:
    """성능 지표 관리 클래스"""
    
    def __init__(self, output_dir: str):
        """지표 관리자 초기화"""
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "performance_metrics.csv")
        
        self.columns = [
            'Cycle', 'Model', 'mAP50', 'Precision', 'Recall', 'F1-Score',
            'Detected_Objects', 'Filtered_Objects', 'Labels_Available'
        ]
        
        if os.path.exists(self.metrics_file):
            try:
                self.metrics_df = pd.read_csv(self.metrics_file)
                if 'Labels_Available' not in self.metrics_df.columns:
                    self.metrics_df['Labels_Available'] = True
            except:
                self.metrics_df = pd.DataFrame(columns=self.columns)
        else:
            self.metrics_df = pd.DataFrame(columns=self.columns)
    
    def add_metrics(self, metrics: Dict):
        """새로운 메트릭 추가"""
        cycle = metrics['Cycle']
        model = metrics['Model']
        
        existing_mask = (self.metrics_df['Cycle'] == cycle) & (self.metrics_df['Model'] == model)
        
        if existing_mask.any():
            for key, value in metrics.items():
                self.metrics_df.loc[existing_mask, key] = value
        else:
            new_row = pd.DataFrame([metrics])
            if self.metrics_df.empty:
                self.metrics_df = new_row
            else:
                self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        
        self.save_metrics()
    
    def save_metrics(self):
        """메트릭 파일 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_df.to_csv(self.metrics_file, index=False)
    
    def get_best_performance(self, metric: str = 'mAP50') -> Dict:
        """최고 성능 반환"""
        if len(self.metrics_df) == 0:
            return None
        
        valid = self.metrics_df[self.metrics_df[metric] >= 0]
        if len(valid) == 0:
            return None
        
        best_idx = valid[metric].idxmax()
        return valid.iloc[best_idx].to_dict()
    
    def export_summary(self, summary_path: str):
        """성능 요약 내보내기"""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Active Learning 성능 요약\n")
            f.write("="*50 + "\n\n")
            
            if len(self.metrics_df) > 0:
                labels_available = self.metrics_df.get('Labels_Available', pd.Series([True])).iloc[0]
                
                if not labels_available:
                    f.write("⚠️ 라벨 없음: 성능 지표 측정 불가\n\n")
                
                f.write("사이클별 성능:\n")
                f.write("-" * 30 + "\n")
                
                for _, row in self.metrics_df.iterrows():
                    cycle = int(row['Cycle'])
                    detected = row['Detected_Objects']
                    filtered = row.get('Filtered_Objects', 0)
                    
                    f.write(f"Cycle {cycle}:\n")
                    f.write(f"  탐지: {detected}개\n")
                    f.write(f"  필터링: {filtered}개\n")
                    
                    if labels_available and row['mAP50'] >= 0:
                        f.write(f"  mAP50: {row['mAP50']:.4f}\n")
                        f.write(f"  Precision: {row['Precision']:.4f}\n")
                        f.write(f"  Recall: {row['Recall']:.4f}\n")
                    else:
                        f.write("  성능: 측정 불가\n")
                    f.write("\n")
                
                if labels_available:
                    best = self.get_best_performance()
                    if best:
                        f.write(f"\n최고 성능:\n")
                        f.write(f"Cycle {int(best['Cycle'])}: mAP50={best['mAP50']:.4f}\n")
