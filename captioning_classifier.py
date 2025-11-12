"""
captioning_classifier.py - 이미지 캡셔닝 기반 객체 분류 모듈 (간소화 버전)
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
import re

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

try:
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    VIT_GPT2_AVAILABLE = True
except ImportError:
    VIT_GPT2_AVAILABLE = False

class ImageCaptioningClassifier:
    """이미지 캡셔닝 기반 객체 분류기"""

    def __init__(self, target_keywords: List[str],
                 model_type: str = "blip",
                 device: torch.device = None,
                 conf_threshold: float = 0.5,
                 gpu_num: int = 0,
                 save_captions: bool = True,
                 captions_dir: str = None):
        """캡셔닝 분류기 초기화"""
        if device is None:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.target_keywords = [k.lower().strip() for k in target_keywords]
        self.model_type = model_type.lower()
        self.conf_threshold = conf_threshold
        self.save_captions = save_captions
        self.captions_dir = captions_dir if captions_dir else "./captions"

        # 캡션 저장용 딕셔너리 (메모리 캐시)
        self.captions_cache: Dict[str, str] = {}

        self.stats = {
            'total_classifications': 0,
            'positive_classifications': 0,
            'negative_classifications': 0,
            'keyword_matches': {},
            'generated_captions': []
        }

        self._initialize_model()

        print(f"캡셔닝 분류기 초기화:")
        print(f"  - 모델: {self.model_type}")
        print(f"  - 키워드: {self.target_keywords}")
    
    def _initialize_model(self):
        """캡셔닝 모델 초기화"""
        try:
            if self.model_type == "blip":
                if not BLIP_AVAILABLE:
                    raise ImportError("transformers 필요")
                
                print("BLIP 모델 로딩...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model.to(self.device)
                self.model.eval()
                print("✓ BLIP 로딩 완료")
                
            elif self.model_type == "vit-gpt2":
                if not VIT_GPT2_AVAILABLE:
                    raise ImportError("VIT-GPT2용 transformers 필요")
                
                print("VIT-GPT2 모델 로딩...")
                model_name = "nlpconnect/vit-gpt2-image-captioning"
                self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
                self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                self.model.to(self.device)
                self.model.eval()
                print("✓ VIT-GPT2 로딩 완료")
            
            else:
                raise ValueError(f"지원하지 않는 모델: {self.model_type}")
                
        except Exception as e:
            print(f"✗ 모델 초기화 실패: {e}")
            self.model = None
    
    def generate_caption(self, image: np.ndarray) -> str:
        """이미지에서 캡션 생성"""
        if self.model is None:
            return ""
        
        try:
            # PIL 이미지 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            if pil_image.size[0] < 10 or pil_image.size[1] < 10:
                return ""
            
            # 캡션 생성
            if self.model_type == "blip":
                return self._generate_blip_caption(pil_image)
            elif self.model_type == "vit-gpt2":
                return self._generate_vit_gpt2_caption(pil_image)
            
            return ""
                
        except Exception as e:
            print(f"⚠️ 캡션 생성 오류: {e}")
            return ""
    
    def _generate_blip_caption(self, pil_image: Image.Image) -> str:
        """BLIP 캡션 생성"""
        try:
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=5)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.lower().strip()
        except Exception as e:
            print(f"⚠️ BLIP 캡션 오류: {e}")
            return ""
    
    def _generate_vit_gpt2_caption(self, pil_image: Image.Image) -> str:
        """VIT-GPT2 캡션 생성"""
        try:
            pixel_values = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True
                )
            
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return caption.lower().strip()
        except Exception as e:
            print(f"⚠️ VIT-GPT2 캡션 오류: {e}")
            return ""
    
    def classify(self, image: np.ndarray, image_name: str = None,
                bbox: Tuple = None, cycle: int = None) -> Tuple[int, float, str]:
        """
        이미지 분류 및 캡션 생성

        Args:
            image: 입력 이미지
            image_name: 이미지 파일명 (캡션 저장용)
            bbox: 바운딩 박스 (x1, y1, x2, y2) (캡션 저장용)
            cycle: 현재 사이클 번호 (캡션 저장용)

        Returns:
            (pred_class, confidence, caption): 0=target, 1=non-target, 신뢰도, 생성된 캡션
        """
        self.stats['total_classifications'] += 1

        if image.shape[0] < 10 or image.shape[1] < 10:
            self.stats['negative_classifications'] += 1
            return 1, 0.0, ""

        try:
            caption = self.generate_caption(image)

            if not caption:
                self.stats['negative_classifications'] += 1
                return 1, 0.0, ""

            # 통계 저장
            self.stats['generated_captions'].append(caption)
            if len(self.stats['generated_captions']) > 100:
                self.stats['generated_captions'] = self.stats['generated_captions'][-100:]

            # 키워드 매칭
            is_positive, matched = self._check_keywords(caption)

            for keyword in matched:
                if keyword not in self.stats['keyword_matches']:
                    self.stats['keyword_matches'][keyword] = 0
                self.stats['keyword_matches'][keyword] += 1

            if is_positive:
                self.stats['positive_classifications'] += 1
                pred_class = 0
                confidence = 1.0
            else:
                self.stats['negative_classifications'] += 1
                pred_class = 1
                confidence = 0.0

            # 캡션 저장 (요청된 경우)
            if self.save_captions and image_name and bbox and cycle is not None:
                self.save_caption(image_name, bbox, caption, pred_class, confidence, cycle)

            return pred_class, confidence, caption

        except Exception as e:
            print(f"⚠️ 분류 오류: {e}")
            self.stats['negative_classifications'] += 1
            return 1, 0.0, ""
    
    def _check_keywords(self, caption: str) -> Tuple[bool, List[str]]:
        """키워드 검사"""
        caption_lower = caption.lower()
        matched = []

        for keyword in self.target_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, caption_lower):
                matched.append(keyword)

        return len(matched) > 0, matched

    def save_caption(self, image_name: str, bbox: Tuple, caption: str,
                    class_id: int, confidence: float, cycle: int):
        """
        객체 캡션을 파일에 저장 (사이클별 분리)

        Args:
            image_name: 이미지 파일명
            bbox: 바운딩 박스 (x1, y1, x2, y2)
            caption: 생성된 캡션
            class_id: 클래스 ID
            confidence: 신뢰도
            cycle: 현재 사이클 번호
        """
        try:
            from utils import append_caption_to_file

            # 사이클별 캡션 디렉토리 생성
            cycle_captions_dir = os.path.join(self.captions_dir, f"cycle_{cycle}")
            os.makedirs(cycle_captions_dir, exist_ok=True)

            # 사이클별 캡션 파일 경로
            captions_file = os.path.join(cycle_captions_dir, "captions.json")

            # 캡션 저장
            append_caption_to_file(
                image_name=image_name,
                bbox=bbox,
                caption=caption,
                class_id=class_id,
                confidence=confidence,
                captions_file=captions_file
            )

            # 캐시에도 저장 (사이클 정보 포함)
            cache_key = f"cycle{cycle}_{image_name}_{bbox}"
            self.captions_cache[cache_key] = caption

        except Exception as e:
            print(f"⚠️ 캡션 저장 중 오류: {e}")

    def get_caption(self, image_name: str, bbox: Tuple) -> str:
        """
        저장된 캡션 조회

        Args:
            image_name: 이미지 파일명
            bbox: 바운딩 박스

        Returns:
            저장된 캡션 (없으면 빈 문자열)
        """
        cache_key = f"{image_name}_{bbox}"
        return self.captions_cache.get(cache_key, "")

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        stats = self.stats.copy()
        if stats['total_classifications'] > 0:
            stats['positive_ratio'] = stats['positive_classifications'] / stats['total_classifications']
        else:
            stats['positive_ratio'] = 0.0
        return stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보"""
        return {
            'model_type': self.model_type,
            'target_keywords': self.target_keywords,
            'device': str(self.device),
            'conf_threshold': self.conf_threshold
        }
    
    def save_model(self, save_path: str):
        """설정 저장"""
        import json
        
        config = {
            'model_type': self.model_type,
            'target_keywords': self.target_keywords,
            'conf_threshold': self.conf_threshold
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        config_path = save_path.replace('.pth', '_config.json')
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"캡셔닝 분류기 설정 저장: {config_path}")

class CaptioningClassifierTrainer:
    """캡셔닝 분류기 트레이너 (설정 관리용)"""
    
    def __init__(self, device: torch.device, **kwargs):
        """트레이너 초기화"""
        self.device = device
        print("캡셔닝 분류기는 사전훈련 모델 사용 (별도 학습 없음)")
    
    def train_classifier(self, cropped_data_dir: str, 
                        previous_model_path: str = None,
                        manual_label_dir: str = None, 
                        cycle: int = 1,
                        target_keywords: List[str] = None, 
                        model_type: str = "blip"):
        """캡셔닝 분류기 생성"""
        if target_keywords is None:
            target_keywords = ['car', 'vehicle', 'automobile']
        
        print(f"Cycle {cycle}용 캡셔닝 분류기 생성:")
        print(f"  - 모델: {model_type}")
        print(f"  - 키워드: {target_keywords}")
        
        try:
            classifier = ImageCaptioningClassifier(
                target_keywords=target_keywords,
                model_type=model_type,
                device=self.device
            )
            print(f"✓ Cycle {cycle}용 캡셔닝 분류기 생성 완료")
            return classifier
        except Exception as e:
            print(f"✗ 캡셔닝 분류기 생성 실패: {e}")
            return None
