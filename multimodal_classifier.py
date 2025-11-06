"""
multimodal_classifier_enhanced.py - Instruction-based VLM을 사용한 멀티모달 필터링 분류 모듈
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image
from tqdm import tqdm
import random

# VLM 관련
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

try:
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    INSTRUCTBLIP_AVAILABLE = True
except ImportError:
    INSTRUCTBLIP_AVAILABLE = False

try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    QWEN_VL_AVAILABLE = True
except ImportError:
    QWEN_VL_AVAILABLE = False

# Accelerate 확인
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

# CNN 관련
from torchvision import models, transforms


class MultimodalAttentionClassifier(nn.Module):
    """멀티모달 어텐션 기반 분류 모델"""
    
    def __init__(self, text_embed_dim=768, image_embed_dim=1024, hidden_dim=512):
        super(MultimodalAttentionClassifier, self).__init__()
        
        # 텍스트 특징 인코더 (VLM 캡션)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 이미지 특징 인코더 (DenseNet)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-Attention 메커니즘
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 분류 헤드
        self.classifier = nn.Linear(256, 2)
        
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: (batch_size, text_embed_dim) - VLM 텍스트 임베딩
            image_features: (batch_size, image_embed_dim) - CNN 이미지 임베딩
        Returns:
            logits: (batch_size, 2) - 클래스 로짓
        """
        # 특징 인코딩
        text_encoded = self.text_encoder(text_features)  # (B, hidden_dim)
        image_encoded = self.image_encoder(image_features)  # (B, hidden_dim)
        
        # 어텐션을 위한 차원 확장
        text_encoded = text_encoded.unsqueeze(1)  # (B, 1, hidden_dim)
        image_encoded = image_encoded.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Cross-Attention: 텍스트를 쿼리로, 이미지를 키/밸류로
        attended_features, _ = self.cross_attention(
            text_encoded, image_encoded, image_encoded
        )  # (B, 1, hidden_dim)
        
        attended_features = attended_features.squeeze(1)  # (B, hidden_dim)
        
        # 특징 융합
        combined = torch.cat([attended_features, image_encoded.squeeze(1)], dim=1)  # (B, hidden_dim*2)
        fused = self.fusion(combined)  # (B, 256)
        
        # 분류
        logits = self.classifier(fused)  # (B, 2)
        
        return logits


class MultimodalDataset(Dataset):
    """멀티모달 학습용 데이터셋"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 text_features: List[np.ndarray], image_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.text_features = text_features
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        
        # 텍스트 특징
        text_feat = torch.FloatTensor(self.text_features[idx])
        
        # 라벨
        label = self.labels[idx]
        
        return image, text_feat, label


class MultimodalFilterClassifier:
    """멀티모달 필터링 분류기 (Instruction-based VLM 지원)"""

    def __init__(self, vlm_model_type: str = "instructblip",
                 target_keywords: List[str] = None,
                 device: torch.device = None,
                 conf_threshold: float = 0.5,
                 gpu_num: int = 0,
                 custom_prompt: str = None,
                 save_captions: bool = True,
                 captions_dir: str = None):
        """
        멀티모달 분류기 초기화

        Args:
            vlm_model_type: VLM 모델 타입 ("blip", "vit-gpt2", "instructblip", "llava", "qwen-vl")
            target_keywords: 타겟 키워드 리스트
            device: 연산 장치
            conf_threshold: 신뢰도 임계값
            gpu_num: GPU 번호
            custom_prompt: 커스텀 instruction prompt (None이면 기본 프롬프트 사용)
            save_captions: 캡션 저장 여부
            captions_dir: 캡션 저장 디렉토리
        """
        if device is None:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.vlm_model_type = vlm_model_type.lower()
        self.target_keywords = target_keywords if target_keywords else ['car']
        self.conf_threshold = conf_threshold
        self.save_captions = save_captions
        self.captions_dir = captions_dir if captions_dir else "./captions"

        # 캡션 저장용 딕셔너리 (메모리 캐시)
        self.captions_cache: Dict[str, str] = {}
        
        # Instruction prompt 설정
        if custom_prompt:
            self.instruction_prompt = custom_prompt
        else:
            # 기본 상세 설명 프롬프트
            keywords_str = ", ".join(self.target_keywords)
            self.instruction_prompt = (
                f"Describe this image in detail. Focus on identifying objects, "
                f"especially {keywords_str}. Include information about: "
                f"1) What objects are present, "
                f"2) Their appearance and characteristics, "
                f"3) Their position and context in the image, "
                f"4) Any distinctive features or attributes."
            )
        
        # VLM 모델 초기화
        self._initialize_vlm()
        
        # CNN 특징 추출기 초기화 (DenseNet121)
        self._initialize_cnn()
        
        # 멀티모달 분류 모델
        self.model = None
        
        print(f"멀티모달 필터링 분류기 초기화:")
        print(f"  - VLM: {self.vlm_model_type}")
        print(f"  - CNN: DenseNet121")
        print(f"  - 키워드: {self.target_keywords}")
        print(f"  - Instruction Prompt: {self.instruction_prompt[:100]}...")
    
    def _initialize_vlm(self):
        """VLM 모델 초기화 (Instruction-based 모델 포함)"""
        try:
            if self.vlm_model_type == "instructblip":
                if not INSTRUCTBLIP_AVAILABLE:
                    raise ImportError("transformers 라이브러리에서 InstructBLIP 필요")

                print("InstructBLIP 모델 로딩...")
                model_name = "Salesforce/instructblip-vicuna-7b"
                self.vlm_processor = InstructBlipProcessor.from_pretrained(model_name)

                # Accelerate 사용 가능 여부에 따라 로딩 방식 결정
                if ACCELERATE_AVAILABLE:
                    self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto"
                    )
                else:
                    print("  ⚠️ Accelerate 미설치 - CPU/단일 GPU 모드로 로딩")
                    self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    self.vlm_model.to(self.device)

                self.vlm_model.eval()
                self.text_embed_dim = 4096  # Vicuna-7B hidden size
                print("✓ InstructBLIP 로딩 완료")
                
            elif self.vlm_model_type == "llava":
                if not LLAVA_AVAILABLE:
                    raise ImportError("transformers 라이브러리에서 LLaVA 필요")

                print("LLaVA 모델 로딩...")
                model_name = "llava-hf/llava-1.5-7b-hf"
                self.vlm_processor = AutoProcessor.from_pretrained(model_name)

                # Accelerate 사용 가능 여부에 따라 로딩 방식 결정
                if ACCELERATE_AVAILABLE:
                    self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto"
                    )
                else:
                    print("  ⚠️ Accelerate 미설치 - CPU/단일 GPU 모드로 로딩")
                    self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    self.vlm_model.to(self.device)

                self.vlm_model.eval()
                self.text_embed_dim = 4096  # LLaMA hidden size
                print("✓ LLaVA 로딩 완료")
                
            elif self.vlm_model_type == "qwen-vl":
                if not QWEN_VL_AVAILABLE:
                    raise ImportError("transformers 라이브러리에서 Qwen-VL 필요")

                print("Qwen-VL 모델 로딩...")
                model_name = "Qwen/Qwen-VL-Chat"
                self.vlm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                # Accelerate 사용 가능 여부에 따라 로딩 방식 결정
                if ACCELERATE_AVAILABLE and torch.cuda.is_available():
                    self.vlm_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    if not ACCELERATE_AVAILABLE:
                        print("  ⚠️ Accelerate 미설치 - CPU/단일 GPU 모드로 로딩")
                    self.vlm_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True
                    )
                    self.vlm_model.to(self.device)

                self.vlm_model.eval()
                self.text_embed_dim = 4096  # Qwen hidden size
                print("✓ Qwen-VL 로딩 완료")
            
            elif self.vlm_model_type == "blip":
                if not BLIP_AVAILABLE:
                    raise ImportError("transformers 필요")
                
                print("BLIP 모델 로딩...")
                self.vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.vlm_model.to(self.device)
                self.vlm_model.eval()
                self.text_embed_dim = 768
                print("✓ BLIP 로딩 완료")
                
            elif self.vlm_model_type == "vit-gpt2":
                if not VIT_GPT2_AVAILABLE:
                    raise ImportError("VIT-GPT2용 transformers 필요")
                
                print("VIT-GPT2 모델 로딩...")
                model_name = "nlpconnect/vit-gpt2-image-captioning"
                self.vlm_model = VisionEncoderDecoderModel.from_pretrained(model_name)
                self.vlm_feature_extractor = ViTImageProcessor.from_pretrained(model_name)
                self.vlm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.vlm_model.to(self.device)
                self.vlm_model.eval()
                self.text_embed_dim = 768
                print("✓ VIT-GPT2 로딩 완료")
            
            else:
                raise ValueError(f"지원하지 않는 VLM 모델: {self.vlm_model_type}")
                
        except Exception as e:
            print(f"✗ VLM 모델 초기화 실패: {e}")
            self.vlm_model = None
    
    def _initialize_cnn(self):
        """CNN 특징 추출기 초기화"""
        print("DenseNet121 특징 추출기 로딩...")
        self.cnn_model = models.densenet121(weights='IMAGENET1K_V1')
        
        # 특징 추출을 위해 분류 헤드 제거
        self.image_embed_dim = self.cnn_model.classifier.in_features
        self.cnn_model.classifier = nn.Identity()
        
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # 이미지 전처리
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✓ DenseNet121 로딩 완료")
    
    def generate_detailed_caption(self, image: np.ndarray, prompt: str = None) -> str:
        """
        Instruction-based VLM을 사용하여 상세한 캡션 생성
        
        Args:
            image: 입력 이미지 (numpy array)
            prompt: 커스텀 프롬프트 (None이면 기본 프롬프트 사용)
        
        Returns:
            생성된 상세 캡션
        """
        if self.vlm_model is None:
            return ""
        
        try:
            # PIL 이미지 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            if pil_image.size[0] < 10 or pil_image.size[1] < 10:
                return ""
            
            instruction = prompt if prompt else self.instruction_prompt
            
            # 모델별 캡션 생성
            if self.vlm_model_type == "instructblip":
                inputs = self.vlm_processor(
                    images=pil_image,
                    text=instruction,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_beams=5,
                        temperature=0.7
                    )
                caption = self.vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            elif self.vlm_model_type == "llava":
                # LLaVA는 대화 형식으로 프롬프트 구성
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction}
                        ]
                    }
                ]
                
                prompt_text = self.vlm_processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True
                )
                
                inputs = self.vlm_processor(
                    images=pil_image,
                    text=prompt_text,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7
                    )
                caption = self.vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            elif self.vlm_model_type == "qwen-vl":
                # Qwen-VL 프롬프트 형식 (이미지를 임시 파일로 저장)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    pil_image.save(tmp_file.name)
                    img_path = tmp_file.name

                try:
                    query = self.vlm_tokenizer.from_list_format([
                        {'image': img_path},
                        {'text': instruction}
                    ])

                    inputs = self.vlm_tokenizer(query, return_tensors='pt').to(self.device)

                    with torch.no_grad():
                        outputs = self.vlm_model.generate(
                            **inputs,
                            max_new_tokens=100
                        )
                    caption = self.vlm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                finally:
                    # 임시 파일 삭제
                    if os.path.exists(img_path):
                        os.remove(img_path)
            
            elif self.vlm_model_type == "blip":
                # BLIP은 instruction을 직접 지원하지 않음 (기본 captioning)
                inputs = self.vlm_processor(pil_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.vlm_model.generate(**inputs, max_new_tokens=50)
                caption = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
            
            elif self.vlm_model_type == "vit-gpt2":
                # VIT-GPT2도 instruction 미지원 (기본 captioning)
                pixel_values = self.vlm_feature_extractor(images=pil_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                with torch.no_grad():
                    outputs = self.vlm_model.generate(pixel_values, max_new_tokens=50)
                caption = self.vlm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            else:
                caption = ""
            
            return caption.strip()
            
        except Exception as e:
            print(f"⚠️ 캡션 생성 오류: {e}")
            return ""
    
    def extract_text_features(self, image: np.ndarray) -> np.ndarray:
        """
        VLM을 사용하여 텍스트 특징 추출 (상세 캡션 기반)
        """
        if self.vlm_model is None:
            return np.zeros(self.text_embed_dim)
        
        try:
            # PIL 이미지 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            if pil_image.size[0] < 10 or pil_image.size[1] < 10:
                return np.zeros(self.text_embed_dim)
            
            # Instruction-based 모델의 경우
            if self.vlm_model_type in ["instructblip", "llava", "qwen-vl"]:
                # 1. 상세 캡션 생성
                caption = self.generate_detailed_caption(image)
                
                # 2. 캡션의 임베딩 추출
                if self.vlm_model_type == "instructblip":
                    inputs = self.vlm_processor(
                        images=pil_image,
                        text=self.instruction_prompt,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        # 인코더의 히든 스테이트 추출
                        outputs = self.vlm_model.vision_model(**inputs.pixel_values)
                        text_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                elif self.vlm_model_type == "llava":
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": self.instruction_prompt}
                            ]
                        }
                    ]
                    prompt_text = self.vlm_processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True
                    )
                    inputs = self.vlm_processor(
                        images=pil_image,
                        text=prompt_text,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.vlm_model.vision_tower(inputs.pixel_values)
                        text_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                elif self.vlm_model_type == "qwen-vl":
                    # Qwen-VL은 이미지를 임시 파일로 저장 필요
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        pil_image.save(tmp_file.name)
                        img_path = tmp_file.name

                    try:
                        query = self.vlm_tokenizer.from_list_format([
                            {'image': img_path},
                            {'text': self.instruction_prompt}
                        ])
                        inputs = self.vlm_tokenizer(query, return_tensors='pt').to(self.device)

                        with torch.no_grad():
                            # 비전 인코더에서 특징 추출
                            outputs = self.vlm_model.transformer(**inputs, output_hidden_states=True)
                            text_features = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
                    finally:
                        if os.path.exists(img_path):
                            os.remove(img_path)
            
            # 기존 captioning 모델의 경우
            elif self.vlm_model_type == "blip":
                inputs = self.vlm_processor(pil_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.vlm_model.generate(**inputs, output_hidden_states=True, return_dict_in_generate=True)
                    if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states:
                        text_features = outputs.decoder_hidden_states[-1][-1].mean(dim=1).squeeze().cpu().numpy()
                    else:
                        encoder_outputs = self.vlm_model.vision_model(**inputs)
                        text_features = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            elif self.vlm_model_type == "vit-gpt2":
                pixel_values = self.vlm_feature_extractor(images=pil_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                with torch.no_grad():
                    encoder_outputs = self.vlm_model.encoder(pixel_values)
                    text_features = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            else:
                return np.zeros(self.text_embed_dim)
            
            return text_features
            
        except Exception as e:
            print(f"⚠️ 텍스트 특징 추출 오류: {e}")
            return np.zeros(self.text_embed_dim)
    
    def extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """CNN을 사용하여 이미지 특징 추출"""
        try:
            # PIL 이미지 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # 전처리 및 특징 추출
            image_tensor = self.image_transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.cnn_model(image_tensor).squeeze().cpu().numpy()
            
            return image_features
            
        except Exception as e:
            print(f"⚠️ 이미지 특징 추출 오류: {e}")
            return np.zeros(self.image_embed_dim)
    
    def classify(self, image: np.ndarray, image_name: str = None,
                bbox: Tuple = None) -> Tuple[int, float, str]:
        """
        객체 이미지 분류 및 캡션 생성

        Args:
            image: 입력 이미지
            image_name: 이미지 파일명 (캡션 저장용)
            bbox: 바운딩 박스 (x1, y1, x2, y2) (캡션 저장용)

        Returns:
            (pred_class, confidence, caption): 0=target, 1=non-target, 신뢰도, 생성된 캡션
        """
        if self.model is None:
            print("⚠️ 모델이 로드되지 않음")
            return 1, 0.0, ""

        if image.shape[0] < 10 or image.shape[1] < 10:
            return 1, 0.0, ""

        try:
            # 캡션 생성
            caption = self.generate_detailed_caption(image)

            # 특징 추출
            text_features = self.extract_text_features(image)
            image_features = self.extract_image_features(image)

            # 텐서 변환
            text_tensor = torch.FloatTensor(text_features).unsqueeze(0).to(self.device)
            image_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(self.device)

            # 분류
            self.model.eval()
            with torch.no_grad():
                logits = self.model(text_tensor, image_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                conf, predicted = torch.max(probabilities, 1)

            pred_class = predicted.item()
            confidence = conf.item()

            # 캡션 저장 (요청된 경우)
            if self.save_captions and image_name and bbox:
                self.save_caption(image_name, bbox, caption, pred_class, confidence)

            return pred_class, confidence, caption

        except Exception as e:
            print(f"⚠️ 분류 중 오류: {e}")
            return 1, 0.0, ""

    def save_caption(self, image_name: str, bbox: Tuple, caption: str,
                    class_id: int, confidence: float):
        """
        객체 캡션을 파일에 저장

        Args:
            image_name: 이미지 파일명
            bbox: 바운딩 박스 (x1, y1, x2, y2)
            caption: 생성된 캡션
            class_id: 클래스 ID
            confidence: 신뢰도
        """
        try:
            from utils import append_caption_to_file

            # 캡션 디렉토리 생성
            os.makedirs(self.captions_dir, exist_ok=True)

            # 캡션 파일 경로
            captions_file = os.path.join(self.captions_dir, "captions.json")

            # 캡션 저장
            append_caption_to_file(
                image_name=image_name,
                bbox=bbox,
                caption=caption,
                class_id=class_id,
                confidence=confidence,
                captions_file=captions_file
            )

            # 캐시에도 저장
            cache_key = f"{image_name}_{bbox}"
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
    
    def load_model(self, model_path: str):
        """학습된 모델 로드"""
        try:
            self.model = MultimodalAttentionClassifier(
                text_embed_dim=self.text_embed_dim,
                image_embed_dim=self.image_embed_dim,
                hidden_dim=512
            )
            
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ 멀티모달 모델 로드 성공: {model_path}")
            return True
            
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
            return False
    
    def save_model(self, save_path: str):
        """모델 저장"""
        if self.model is None:
            print("⚠️ 저장할 모델이 없음")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"멀티모달 모델 저장: {save_path}")
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'vlm_type': self.vlm_model_type,
            'cnn_type': 'DenseNet121',
            'text_embed_dim': self.text_embed_dim,
            'image_embed_dim': self.image_embed_dim,
            'target_keywords': self.target_keywords,
            'device': str(self.device),
            'conf_threshold': self.conf_threshold,
            'instruction_prompt': self.instruction_prompt
        }


class MultimodalFilterTrainer:
    """멀티모달 필터링 모델 학습 관리자"""
    
    def __init__(self, device: torch.device,
                 batch_size: int = 16,
                 num_epochs: int = 20,
                 learning_rate: float = 0.001):
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
    
    def collect_training_data(self, image_dir: str, label_dir: str,
                            detector, cycle: int,
                            num_samples_per_class: int = 100,
                            iou_threshold: float = 0.5) -> Tuple[List[str], List[int]]:
        """
        첫 사이클에서 IoU 기준으로 올바른 target/non-target 샘플 수집
        
        Returns:
            (image_paths, labels): 수집된 이미지 경로와 라벨 리스트
        """
        from utils import get_image_files, load_yolo_labels, yolo_to_xyxy, calculate_iou
        
        print(f"\n{'='*60}")
        print(f"Cycle {cycle}: 멀티모달 학습 데이터 수집")
        print(f"{'='*60}")
        
        target_samples = []  # (image_path, class_label)
        nontarget_samples = []
        
        image_files = get_image_files(image_dir)
        
        # 임시 저장 디렉토리
        temp_dir = f"/tmp/multimodal_training_data_cycle{cycle}"
        target_dir = os.path.join(temp_dir, "target")
        nontarget_dir = os.path.join(temp_dir, "nontarget")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(nontarget_dir, exist_ok=True)
        
        for image_path in tqdm(image_files, desc="IoU 기반 샘플 수집"):
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # GT 라벨 로드
            base_name = os.path.basename(image_path)
            label_name = os.path.splitext(base_name)[0] + '.txt'
            gt_label_path = os.path.join(label_dir, label_name)
            
            if not os.path.exists(gt_label_path):
                continue
            
            gt_objects = load_yolo_labels(gt_label_path)
            if len(gt_objects) == 0:
                continue
            
            gt_boxes = [yolo_to_xyxy(gt_obj, w, h) for gt_obj in gt_objects]
            
            # 객체 탐지
            results = detector.model.predict(
                source=img,
                conf=detector.conf_threshold,
                iou=detector.iou_threshold,
                save=False,
                verbose=False
            )
            
            if len(results[0].boxes) == 0:
                continue
            
            # 각 탐지된 객체에 대해 IoU 계산
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_box = (x1, y1, x2, y2)
                
                # 크롭
                if x1 >= x2 or y1 >= y2:
                    continue
                
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                obj_img = img[int(y1):int(y2), int(x1):int(x2)]
                
                if obj_img.size == 0 or obj_img.shape[0] < 10 or obj_img.shape[1] < 10:
                    continue
                
                # GT와의 최대 IoU 계산
                max_iou = 0.0
                for gt_box in gt_boxes:
                    iou = calculate_iou(pred_box, gt_box)
                    max_iou = max(max_iou, iou)
                
                # IoU 기준으로 분류
                if max_iou >= iou_threshold:
                    # Target (올바른 탐지)
                    if len(target_samples) < num_samples_per_class:
                        obj_filename = f"{os.path.splitext(base_name)[0]}_obj{idx}_iou{max_iou:.3f}.jpg"
                        obj_path = os.path.join(target_dir, obj_filename)
                        cv2.imwrite(obj_path, obj_img)
                        target_samples.append((obj_path, 0))
                else:
                    # Non-target (잘못된 탐지 또는 배경)
                    if len(nontarget_samples) < num_samples_per_class:
                        obj_filename = f"{os.path.splitext(base_name)[0]}_obj{idx}_iou{max_iou:.3f}.jpg"
                        obj_path = os.path.join(nontarget_dir, obj_filename)
                        cv2.imwrite(obj_path, obj_img)
                        nontarget_samples.append((obj_path, 1))
                
                # 충분한 샘플 수집 시 종료
                if (len(target_samples) >= num_samples_per_class and 
                    len(nontarget_samples) >= num_samples_per_class):
                    break
            
            if (len(target_samples) >= num_samples_per_class and 
                len(nontarget_samples) >= num_samples_per_class):
                break
        
        # 결과 결합
        all_samples = target_samples + nontarget_samples
        random.shuffle(all_samples)
        
        image_paths = [s[0] for s in all_samples]
        labels = [s[1] for s in all_samples]
        
        print(f"수집 완료:")
        print(f"  - Target: {len(target_samples)}개")
        print(f"  - Non-target: {len(nontarget_samples)}개")
        print(f"  - 총: {len(all_samples)}개")
        
        return image_paths, labels
    
    def train_model(self, classifier: MultimodalFilterClassifier,
                   image_paths: List[str], labels: List[int],
                   save_dir: str) -> bool:
        """
        멀티모달 모델 학습
        
        Args:
            classifier: 멀티모달 분류기 객체
            image_paths: 학습 이미지 경로 리스트
            labels: 라벨 리스트
            save_dir: 모델 저장 디렉토리
            
        Returns:
            성공 여부
        """
        if len(image_paths) < 10:
            print("⚠️ 학습 데이터 부족 (최소 10개 필요)")
            return False
        
        print(f"\n{'='*60}")
        print("멀티모달 모델 학습 시작")
        print(f"{'='*60}")
        print(f"학습 샘플 수: {len(image_paths)}")
        
        # 1. 특징 추출
        print("\n[1/3] 특징 추출 중...")
        text_features = []
        image_features_list = []
        valid_indices = []
        
        for idx, img_path in enumerate(tqdm(image_paths, desc="특징 추출")):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            text_feat = classifier.extract_text_features(img)
            img_feat = classifier.extract_image_features(img)
            
            text_features.append(text_feat)
            image_features_list.append(img_feat)
            valid_indices.append(idx)
        
        # 유효한 샘플만 선택
        valid_labels = [labels[i] for i in valid_indices]
        
        print(f"유효 샘플: {len(valid_indices)}개")
        
        if len(valid_indices) < 10:
            print("⚠️ 유효한 학습 데이터 부족")
            return False
        
        # 2. 데이터셋 준비
        print("\n[2/3] 데이터셋 준비 중...")
        
        # 학습/검증 분할
        split_idx = int(len(valid_indices) * 0.8)
        
        train_text_feats = text_features[:split_idx]
        train_img_feats = image_features_list[:split_idx]
        train_labels = valid_labels[:split_idx]
        
        val_text_feats = text_features[split_idx:]
        val_img_feats = image_features_list[split_idx:]
        val_labels = valid_labels[split_idx:]
        
        print(f"학습 샘플: {len(train_labels)}개")
        print(f"검증 샘플: {len(val_labels)}개")
        
        # 3. 모델 학습
        print("\n[3/3] 모델 학습 중...")
        
        # 멀티모달 모델 초기화
        classifier.model = MultimodalAttentionClassifier(
            text_embed_dim=classifier.text_embed_dim,
            image_embed_dim=classifier.image_embed_dim,
            hidden_dim=512
        ).to(self.device)
        
        # 옵티마이저 및 손실 함수
        optimizer = optim.Adam(classifier.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 텐서 변환
        train_text_tensor = torch.FloatTensor(np.array(train_text_feats))
        train_img_tensor = torch.FloatTensor(np.array(train_img_feats))
        train_label_tensor = torch.LongTensor(train_labels)
        
        val_text_tensor = torch.FloatTensor(np.array(val_text_feats))
        val_img_tensor = torch.FloatTensor(np.array(val_img_feats))
        val_label_tensor = torch.LongTensor(val_labels)
        
        # 학습 루프
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            classifier.model.train()
            
            # 미니배치 학습
            num_batches = (len(train_labels) + self.batch_size - 1) // self.batch_size
            total_loss = 0.0
            correct = 0
            total = 0
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_labels))
                
                batch_text = train_text_tensor[start_idx:end_idx].to(self.device)
                batch_img = train_img_tensor[start_idx:end_idx].to(self.device)
                batch_labels = train_label_tensor[start_idx:end_idx].to(self.device)
                
                optimizer.zero_grad()
                outputs = classifier.model(batch_text, batch_img)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            train_acc = correct / total
            avg_loss = total_loss / num_batches
            
            # 검증
            classifier.model.eval()
            with torch.no_grad():
                val_text = val_text_tensor.to(self.device)
                val_img = val_img_tensor.to(self.device)
                val_labels_dev = val_label_tensor.to(self.device)
                
                val_outputs = classifier.model(val_text, val_img)
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = (val_predicted == val_labels_dev).sum().item() / len(val_labels)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = classifier.model.state_dict().copy()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}: "
                      f"Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # 최고 모델 로드
        if best_model_state:
            classifier.model.load_state_dict(best_model_state)
            classifier.model.eval()
            print(f"\n✓ 학습 완료: 최고 검증 정확도 {best_val_acc:.4f}")
            
            # 모델 저장
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "multimodal_filter.pth")
            classifier.save_model(model_path)
            
            return True
        
        return False


# 사용 예시
if __name__ == "__main__":
    # Instruction-based VLM 사용 예시
    
    # 1. InstructBLIP 사용
    classifier_instructblip = MultimodalFilterClassifier(
        vlm_model_type="instructblip",
        target_keywords=["car", "vehicle"],
        custom_prompt=(
            "Analyze this image and provide a detailed description. "
            "Focus on identifying any vehicles, especially cars. "
            "Describe their type, color, position, and any distinctive features."
        )
    )
    
    # 2. LLaVA 사용
    classifier_llava = MultimodalFilterClassifier(
        vlm_model_type="llava",
        target_keywords=["car"],
        custom_prompt=(
            "Describe what you see in this image with special attention to vehicles. "
            "Include details about: object types, colors, positions, and context."
        )
    )
    
    # 3. 테스트 이미지로 상세 캡션 생성
    import cv2
    test_image = cv2.imread("test_car.jpg")
    
    if test_image is not None:
        # 상세 캡션 생성
        caption = classifier_instructblip.generate_detailed_caption(test_image)
        print(f"Generated Caption: {caption}")
        
        # 분류
        pred_class, confidence = classifier_instructblip.classify(test_image)
        print(f"Classification: {'Target' if pred_class == 0 else 'Non-target'} (conf: {confidence:.3f})")