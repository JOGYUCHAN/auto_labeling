"""
classifier.py - 객체 분류 모델 모듈 (간소화 버전)
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Optional
import numpy as np

class ClassificationDataset(Dataset):
    """분류 모델 학습용 데이터셋"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ObjectClassifier:
    """객체 분류 모델 클래스"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: Optional[torch.device] = None, 
                 conf_threshold: float = 0.5, 
                 gpu_num: int = 0):
        """분류 모델 초기화"""
        if device is None:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.conf_threshold = conf_threshold
        self.model = models.densenet121(weights=None)
        
        # 분류기 헤드 설정
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 2)
        )
        
        # 가중치 로드
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_weights(self, model_path: str):
        """모델 가중치 로딩"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✓ 분류 모델 로딩 성공: {model_path}")
        except Exception as e:
            print(f"⚠️ 모델 로딩 실패: {e}")
    
    def classify(self, image: np.ndarray) -> Tuple[int, float]:
        """객체 이미지 분류"""
        if image.shape[0] < 10 or image.shape[1] < 10:
            return 1, 0.0
        
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), conf.item()
        except Exception as e:
            print(f"⚠️ 분류 중 오류: {e}")
            return 1, 0.0
    
    def save_model(self, save_path: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"분류 모델 저장: {save_path}")
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'conf_threshold': self.conf_threshold
        }

class ClassifierTrainer:
    """분류 모델 학습 관리 클래스"""
    
    def __init__(self, device: torch.device, 
                 max_samples_per_class: int = 100,
                 batch_size: int = 16, 
                 num_epochs: int = 15,
                 lr_new: float = 0.001, 
                 lr_finetune: float = 0.0001):
        """분류 모델 트레이너 초기화"""
        self.device = device
        self.max_samples_per_class = max_samples_per_class
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_new = lr_new
        self.lr_finetune = lr_finetune
    
    def train_classifier(self, cropped_data_dir: str, 
                        previous_model_path: Optional[str] = None,
                        manual_label_dir: Optional[str] = None, 
                        cycle: int = 1) -> Optional[ObjectClassifier]:
        """분류 모델 학습"""
        from utils import get_image_files, set_seed
        
        set_seed()
        
        # 데이터 수집
        class0_dir = os.path.join(cropped_data_dir, "class0")
        class1_dir = os.path.join(cropped_data_dir, "class1")
        
        class0_paths = get_image_files(class0_dir)
        class1_paths = get_image_files(class1_dir)
        
        # 수동 라벨링 데이터 추가
        if cycle <= 2 and manual_label_dir:
            manual_class0 = os.path.join(manual_label_dir, "class0")
            manual_class1 = os.path.join(manual_label_dir, "class1")
            
            if os.path.exists(manual_class0):
                class0_paths.extend(get_image_files(manual_class0))
            if os.path.exists(manual_class1):
                class1_paths.extend(get_image_files(manual_class1))
        
        print(f"학습 데이터: class0={len(class0_paths)}, class1={len(class1_paths)}")
        
        # 데이터 부족 체크
        if len(class0_paths) < 5 or len(class1_paths) < 5:
            print(f"⚠️ 학습 데이터 부족 (각 클래스당 최소 5개 필요)")
            return None
        
        # 데이터 균형 조정
        class0_paths, class1_paths = self._balance_data(class0_paths, class1_paths)
        
        # 데이터셋 준비
        all_paths = class0_paths + class1_paths
        all_labels = [0] * len(class0_paths) + [1] * len(class1_paths)
        
        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        all_paths, all_labels = zip(*combined)
        
        # 학습/검증 분할
        split_idx = int(len(all_paths) * 0.8)
        train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
        train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
        
        # 모델 초기화
        is_fine_tuning = previous_model_path and os.path.exists(previous_model_path)
        
        if is_fine_tuning:
            print(f"파인튜닝: {previous_model_path}")
            classifier_model = ObjectClassifier(model_path=previous_model_path, device=self.device)
        else:
            if not previous_model_path:
                raise ValueError("초기 가중치가 필요합니다")
            print(f"새 학습: {previous_model_path}")
            classifier_model = ObjectClassifier(model_path=previous_model_path, device=self.device)
        
        # 학습 실행
        return self._train_model(classifier_model, train_paths, train_labels, 
                                val_paths, val_labels, is_fine_tuning)
    
    def _balance_data(self, class0_paths: List[str], 
                     class1_paths: List[str]) -> Tuple[List[str], List[str]]:
        """데이터 균형 조정"""
        # 최대 샘플 수 제한
        if len(class0_paths) > self.max_samples_per_class:
            class0_paths = random.sample(class0_paths, self.max_samples_per_class)
        if len(class1_paths) > self.max_samples_per_class:
            class1_paths = random.sample(class1_paths, self.max_samples_per_class)
        
        # 불균형 조정
        min_samples = min(len(class0_paths), len(class1_paths))
        imbalance_ratio = max(len(class0_paths), len(class1_paths)) / min_samples if min_samples > 0 else 1.0
        
        if imbalance_ratio >= 1.5:
            if len(class0_paths) > len(class1_paths):
                class0_paths = random.sample(class0_paths, min_samples)
            else:
                class1_paths = random.sample(class1_paths, min_samples)
        
        print(f"균형 조정 후: class0={len(class0_paths)}, class1={len(class1_paths)}")
        return class0_paths, class1_paths
    
    def _train_model(self, classifier_model: ObjectClassifier, 
                    train_paths: List[str], train_labels: List[int],
                    val_paths: List[str], val_labels: List[int], 
                    is_fine_tuning: bool) -> Optional[ObjectClassifier]:
        """모델 학습 실행"""
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = ClassificationDataset(train_paths, train_labels, transform=data_transform)
        val_dataset = ClassificationDataset(val_paths, val_labels, transform=data_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=2)
        
        criterion = nn.CrossEntropyLoss()
        lr = self.lr_finetune if is_fine_tuning else self.lr_new
        optimizer = optim.Adam(classifier_model.model.parameters(), lr=lr)
        
        best_acc = 0.0
        best_model_state = None
        
        classifier_model.model.train()
        
        for epoch in range(self.num_epochs):
            running_corrects = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = classifier_model.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
            
            train_acc = running_corrects.double() / len(train_dataset)
            
            # 검증
            classifier_model.model.eval()
            val_corrects = 0
            
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs = val_inputs.to(self.device)
                    val_labels = val_labels.to(self.device)
                    
                    val_outputs = classifier_model.model(val_inputs)
                    _, val_preds = torch.max(val_outputs, 1)
                    val_corrects += torch.sum(val_preds == val_labels.data)
            
            val_acc = val_corrects.double() / len(val_dataset)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = classifier_model.model.state_dict().copy()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}: Train={train_acc:.4f}, Val={val_acc:.4f}")
            
            classifier_model.model.train()
        
        if best_model_state:
            classifier_model.model.load_state_dict(best_model_state)
            classifier_model.model.eval()
            print(f"✓ 학습 완료: 최고 정확도 {best_acc:.4f}")
            return classifier_model
        
        return None
