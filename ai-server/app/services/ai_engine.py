import gc

import scipy.io.wavfile
import torch

from app.emotion.main_library import emotion, emotion_to_music_prompt, evict_emotion_models, _MODELS


class AIEngine:
    """
    AI Model Lifecycle & Inference Manager

    VRAM을 두 페이즈로 분리하여 피크 사용량을 최소화:
      Phase 1 — 감정 분석: YOLO/EMOTIC/CLIP 각각이 추론 직후 CPU로 자체 반환
      Phase 2 — 음악 생성: 감정 모델 완전 퇴거 후 MusicGen 단독으로 GPU 점유
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, local_image_path: str, local_output_path: str, duration: int = 10) -> dict:
        try:
            # ── Phase 1: 감정 분석 ─────────────────────────────────────────
            # CLIP: predict_emotions() finally에서 자체 CPU 반환
            # EMOTIC: analyze_emotions() finally에서 자체 CPU 반환
            # YOLO: yolo() finally에서 자체 CPU 반환
            print(f"[AI Engine] Phase 1 — Emotion analysis: {local_image_path}")
            emotion_result = emotion(local_image_path, 0.5, 5)
            emotion_label = emotion_result["emotion"]
            caption = emotion_result["caption"]

            # Phase 전환: 감정 모델 완전 퇴거 확인 + 캐시 정리
            evict_emotion_models()

            # ── Phase 2: 음악 생성 ─────────────────────────────────────────
            prompt = emotion_to_music_prompt(emotion_label, caption)
            print(f"[AI Engine] Phase 2 — Music generation, prompt: {prompt}")

            _MODELS["music_model"].to(self.device)

            processor = _MODELS["music_processor"]
            model = _MODELS["music_model"]

            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=int(duration * 50),
                    do_sample=True,
                    temperature=1.0,
                    top_k=250,
                    top_p=0.95,
                )

            sampling_rate = model.config.audio_encoder.sampling_rate
            # fp16 모델 출력을 scipy 호환 float32로 변환
            audio_data = audio_values[0, 0].cpu().float().numpy()

            scipy.io.wavfile.write(local_output_path, rate=sampling_rate, data=audio_data)
            print(f"[AI Engine] Audio saved to {local_output_path}")

            return {
                "emotion": emotion_label,
                "caption": caption,
                "prompt": prompt
            }

        except Exception as e:
            print(f"[AI Engine] Generation Error: {e}")
            raise

        finally:
            # MusicGen CPU 반환 + 캐시 정리
            if _MODELS.get("music_model") is not None:
                _MODELS["music_model"].to("cpu")
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("[AI Engine] VRAM cleared.")


ai_engine = AIEngine()
