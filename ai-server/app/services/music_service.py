import asyncio
import gc
import torch
from fastapi.concurrency import run_in_threadpool

from app.emotion.main_library import emotion, emotion_to_music_prompt, _MODELS
from app.utils.s3 import upload_audio, download_image_from_s3, generate_presigned_url

gpu_semaphore = asyncio.Semaphore(1)


class AIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_permanent_models()

    def load_permanent_models(self):
        if _MODELS.get("music_model"):
            print(f"[AI Engine] Permanent Load: MusicGen to {self.device}")
            _MODELS["music_model"].to(self.device)
            _MODELS["music_model"].eval()

    def purge_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[AI Engine] VRAM Purged. Target: ~60% (4.8GB) Stability.")

    async def run_preprocessing_on_demand(self, image_path):
        """
        [On-demand Loading] 인물/자연물 판별 및 전처리 모델을 요청 시점에만 로드합니다.
        """
        try:
            result = await run_in_threadpool(emotion, image_path, 0.5, 5)
            return result
        finally:
            self.purge_vram()

    async def generate_music_pipeline(self, s3_key: str, duration: int = 10) -> dict:
        async with gpu_semaphore:
            try:
                image_path = await download_image_from_s3(s3_key)

                emotion_result = await self.run_preprocessing_on_demand(image_path)
                emotion_label = emotion_result["emotion"]
                caption = emotion_result["caption"]

                prompt = emotion_to_music_prompt(emotion_label, caption)

                processor = _MODELS["music_processor"]
                model = _MODELS["music_model"]

                def _generate():
                    inputs = processor(text=[prompt], return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        # max_new_tokens 등을 조절하여 생성 시간 및 자원 사용량 최적화
                        audio_values = model.generate(**inputs, max_new_tokens=int(duration * 50))
                    return audio_values[0, 0].detach().cpu().numpy()

                audio_data = await run_in_threadpool(_generate)

                # 결과 저장 및 반환
                audio_s3_key = await upload_audio(audio_data, model.config.audio_encoder.sampling_rate)
                presigned_url = await generate_presigned_url(audio_s3_key)

                return {
                    "s3_key": audio_s3_key,
                    "url": presigned_url,
                    "emotion": emotion_label,
                    "caption": caption,
                    "prompt": prompt
                }

            except Exception as e:
                print(f"[AI Engine] Pipeline Error: {e}")
                raise e
            finally:
                # 잔여 캐시 정리
                self.purge_vram()


ai_engine = AIEngine()