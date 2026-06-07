import gc
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import torch

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from app.emotion.color_inference import ColorEmotionInference
    COLOR_AVAILABLE = True
except ImportError:
    COLOR_AVAILABLE = False

try:
    from app.emotion.clip_inference import CLIPEmotionInference
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from app.emotion.face_inference import FaceEmotionAnalyzer
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False

try:
    from app.emotion.moondream_caption import MoondreamCaptioner
    MOONDREAM_AVAILABLE = True
except ImportError:
    MOONDREAM_AVAILABLE = False

try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    MUSICGEN_AVAILABLE = True
except ImportError:
    MUSICGEN_AVAILABLE = False

_MODELS = {
    'yolo': None,
    'color_analyzer': None,
    'face_analyzer': None,  
    'clip_analyzer': None,
    'captioner': None,
    'music_processor': None,
    'music_model': None,
    'music_device': None
}

EMOTIONS = [
    "Happiness", "Confidence", "Surprise", "Pain", "Disquietment",
    "Fear", "Yearning", "Excitement", "Embarrassment", "Affection",
    "Aversion", "Engagement", "Anticipation", "Sensitivity",
    "Annoyance", "Sympathy", "Pleasure"
]

# 1. 모델 로드
def model_load(
    yolo_model_path: str = "model/yolo/yolov8s.pt",
    clip_model_path: str = "model/clip/mlp.pt",
    moondream_model_path: Optional[str] = None,
    face_experiment_path: str = "model",
    face_model_dir: str = "emotic",
    verbose: bool = True
) -> bool:

    if verbose: print("시스템 초기화 중 (VRAM 최적화 모드)...")

    _load_musicgen_model(verbose)

    try:
        if YOLO_AVAILABLE:
            _MODELS['yolo'] = YOLO(yolo_model_path)

        if COLOR_AVAILABLE:
            _MODELS['color_analyzer'] = ColorEmotionInference()

        if FACE_AVAILABLE:
            _MODELS['face_analyzer'] = FaceEmotionAnalyzer(
                experiment_path=face_experiment_path,
                model_dir=face_model_dir
            )

        if CLIP_AVAILABLE:
            # 수정된 CLIPEmotionInference (내부적으로 CPU 대기 로직 포함)
            _MODELS['clip_analyzer'] = CLIPEmotionInference(clip_model_path)

        if MOONDREAM_AVAILABLE:
            _MODELS['captioner'] = MoondreamCaptioner(moondream_model_path)

        return True
    except Exception as e:
        print(f"초기 로드 실패: {e}")
        return False

# 2. YOLO 사람 탐지
def yolo(image_path: str, confidence: float = 0.5) -> Dict:
    if _MODELS['yolo'] is None:
        return {"has_person": False, "error": "YOLO 모델이 로드되지 않음"}

    try:
        results = _MODELS['yolo'](image_path, verbose=False)
        person_detections, has_person = [], False

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if class_id == 0 and conf >= confidence:  # person
                        has_person = True
                        bbox = box.xyxy[0].cpu().numpy()
                        person_detections.append({
                            "bbox": bbox.tolist(),
                            "confidence": conf,
                            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        })

        return {
            "has_person": has_person,
            "person_count": len(person_detections),
            "detections": person_detections,
            "confidence_threshold": confidence
        }
    except Exception as e:
        return {"has_person": False, "error": str(e)}
    finally:
        # YOLO 추론 후 즉시 CPU로 반환
        if torch.cuda.is_available() and hasattr(_MODELS['yolo'], 'model'):
            _MODELS['yolo'].model.to('cpu')
            torch.cuda.empty_cache()

def moondream2(image_path: str) -> str:
    """이미지 캡션 생성"""
    if _MODELS['captioner'] is None:
        return "Caption generation not available"
    
    try:
        return _MODELS['captioner'].generate_caption(image_path)
    except Exception as e:
        print(f"캡션 생성 실패: {e}")
        return "Caption generation failed"

def emotic(image_path: str, person_detections: List[Dict]) -> Optional[Dict]:
    if _MODELS['face_analyzer'] is None:
        return None
    
    try:
        return _MODELS['face_analyzer'].analyze_emotions(image_path, person_detections)
    except Exception as e:
        print(f"얼굴 감정 분석 실패: {e}")
        return None

# 4.2. 색상 감정 추출
def color(image_path: str, n_colors: int = 5) -> Optional[Dict]:
    if _MODELS['color_analyzer'] is None:
        return None
    
    try:
        return _MODELS['color_analyzer'].predict_emotions(image_path, n_colors=n_colors)
    except Exception as e:
        print(f"색상 감정 분석 실패: {e}")
        return None

def caption(image_path: str, caption_text: str) -> Optional[Dict]:
    if _MODELS['clip_analyzer'] is None or not caption_text or caption_text == "Caption generation not available":
        return None
    
    try:
        return _MODELS['clip_analyzer'].predict_emotions(image_path, caption=caption_text)
    except Exception as e:
        print(f"캡션 감정 분석 실패: {e}")
        return None

def evict_emotion_models() -> None:
    """Phase 전환 전 감정 분석 모델 전체를 CPU로 내리는 안전망."""
    if not torch.cuda.is_available():
        return

    m = _MODELS.get('yolo')
    if m is not None and hasattr(m, 'model'):
        m.model.to('cpu')

    m = _MODELS.get('clip_analyzer')
    if m is not None:
        if hasattr(m, 'clip_model') and m.clip_model is not None:
            m.clip_model.to('cpu')
        if hasattr(m, 'mlp_model') and m.mlp_model is not None:
            m.mlp_model.to('cpu')

    m = _MODELS.get('face_analyzer')
    if m is not None:
        if hasattr(m, 'models') and m.models is not None:
            for sub in m.models:
                if hasattr(sub, 'to'):
                    sub.to('cpu')
        if hasattr(m, 'yolo') and m.yolo is not None:
            m.yolo.to('cpu')

    m = _MODELS.get('captioner')
    if m is not None and hasattr(m, 'model') and m.model is not None:
        m.model.to('cpu')

    gc.collect()
    torch.cuda.empty_cache()


def integrate_emotions(
    emotic_result: Optional[Dict] = None,
    color_result: Optional[Dict] = None,
    caption_result: Optional[Dict] = None,
    has_person: bool = True
) -> str:
    
    if has_person:
        # 사람이 있을 때: 얼굴(0.5) + 캡션(0.3) + 색상(0.2)
        face_weight = 0.5 if emotic_result and 'error' not in emotic_result else 0.0
        caption_weight = 0.3 if caption_result and 'error' not in caption_result else 0.0
        color_weight = 0.2 if color_result and 'error' not in color_result else 0.0
    else:
        # 사람이 없을 때: 색상(0.6) + 캡션(0.4)
        face_weight = 0.0
        caption_weight = 0.4 if caption_result and 'error' not in caption_result else 0.0
        color_weight = 0.6 if color_result and 'error' not in color_result else 0.0
    
    total_weight = face_weight + caption_weight + color_weight
    if total_weight == 0:
        return "Unknown"
    
    # 정규화
    face_weight /= total_weight
    caption_weight /= total_weight  
    color_weight /= total_weight
    
    # 감정 점수 통합
    integrated_scores = {}
    for emotion in EMOTIONS:
        score = 0.0
        if face_weight and emotic_result:
            score += face_weight * emotic_result.get("emotion_percentages", {}).get(emotion, 0.0)
        if caption_weight and caption_result:
            score += caption_weight * caption_result.get("emotion_percentages", {}).get(emotion, 0.0)
        if color_weight and color_result:
            score += color_weight * color_result.get("emotion_percentages", {}).get(emotion, 0.0)
        integrated_scores[emotion] = score
    
    # 가장 높은 감정 반환
    if integrated_scores:
        top_emotion = max(integrated_scores.items(), key=lambda x: x[1])
        return top_emotion[0]
    
    return "Unknown"

def emotion(image_path: str, confidence: float = 0.5, n_colors: int = 5) -> Dict:

    print(f"--- 분석 파이프라인 가동: {Path(image_path).name} ---")

    person_detection = yolo(image_path, confidence)
    has_person = person_detection.get("has_person", False)

    caption_text = moondream2(image_path)

    emotic_result = None
    if has_person:
        emotic_result = emotic(image_path, person_detection.get("detections", []))

    color_result = color(image_path, n_colors)  # CPU 연산 (K-means)
    caption_result = caption(image_path, caption_text)  # CLIP GPU On-demand

    final_emotion = integrate_emotions(emotic_result, color_result, caption_result, has_person)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "emotion": final_emotion,
        "caption": caption_text,
        "has_person": has_person
    }

def _load_musicgen_model(verbose: bool = True):
    """MusicGen을 GPU에 영구 적재하는 전용 함수"""
    if _MODELS.get('music_model') is not None:
        return True

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS['music_processor'] = AutoProcessor.from_pretrained("facebook/musicgen-small")

        _MODELS['music_model'] = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small", torch_dtype=torch.float16
        ).to(device)
        _MODELS['music_device'] = device
        _MODELS['music_model'].eval()
        return True
    except Exception as e:
        print(f"MusicGen 상시 적재 실패: {e}")
        return False

def emotion_to_music_prompt(emotion: str, caption: str = "") -> str:
    # 기본 감정 프롬프트
    base_prompt = f"music expressing {emotion.lower()} emotion"
    
    # 캡션이 있으면 추가
    if caption and caption not in ["Caption generation not available", "Caption generation failed"]:
        full_prompt = f"{base_prompt}, inspired by: {caption}"
    else:
        full_prompt = base_prompt
    
    return full_prompt


def generate_music(emotion_text: str, caption: str = "", max_duration: int = 10, verbose: bool = True) -> Dict:
    if _MODELS['music_model'] is None:
        return {"success": False, "error": "MusicGen Not Ready"}

    prompt = emotion_to_music_prompt(emotion_text, caption)

    try:
        device = _MODELS['music_device']
        inputs = _MODELS['music_processor'](text=[prompt], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            audio_values = _MODELS['music_model'].generate(**inputs, max_new_tokens=500)

        audio_data = audio_values[0, 0].detach().cpu().numpy()
        sampling_rate = _MODELS['music_model'].config.audio_encoder.sampling_rate

        return {
            "success": True,
            "audio_data": audio_data,
            "sampling_rate": sampling_rate,
            "prompt": prompt
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def initialize_system(**kwargs) -> bool:
    return model_load(**kwargs)

def analyze_image_emotion(image_path: str, **kwargs) -> str:
    result = emotion(image_path, **kwargs)
    return result["emotion"]
