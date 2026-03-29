import colorsys
from typing import List, Dict, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColorEmotionInference:
    """색상 기반 감정 추론 모듈"""

    def __init__(self):
        """
        색상 기반 감정 추론기
        색채 심리학 이론을 기반으로 한 색상-감정 매핑
        """
        # 17가지 감정 정의 (CLIP 모델과 동일)
        self.emotions = [
            "Happiness", "Confidence", "Surprise", "Pain", "Disquietment",
            "Fear", "Yearning", "Excitement", "Embarrassment", "Affection",
            "Aversion", "Engagement", "Anticipation", "Sensitivity",
            "Annoyance", "Sympathy", "Pleasure"
        ]

        # 색상-감정 매핑 (퍼센트 기반)
        self.color_emotion_mapping = self._initialize_color_emotion_mapping()

    def _initialize_color_emotion_mapping(self) -> Dict:
        """
        색상과 감정 간의 매핑 정의 (퍼센트 기반) - 더 구체적이고 세밀한 분류
        HSV 색상 공간 기반으로 정의
        """
        return {
            # === 빨간색 계열 ===
            "pure_red": {
                "emotions": {
                    "Excitement": 0.850, "Confidence": 0.780, "Annoyance": 0.650,
                    "Pain": 0.520, "Engagement": 0.480
                },
                "hue_range": [(0, 15), (345, 360)], "saturation_min": 0.6, "value_min": 0.4
            },
            "dark_red": {
                "emotions": {
                    "Pain": 0.720, "Annoyance": 0.680, "Aversion": 0.620,
                    "Disquietment": 0.580, "Fear": 0.450
                },
                "hue_range": [(0, 15), (345, 360)], "saturation_min": 0.4, "value_max": 0.4
            },
            "light_red": {
                "emotions": {
                    "Affection": 0.680, "Excitement": 0.620, "Happiness": 0.580,
                    "Embarrassment": 0.540, "Pleasure": 0.480
                },
                "hue_range": [(0, 15), (345, 360)], "saturation_range": (0.3, 0.6), "value_min": 0.7
            },

            # === 주황색 계열 ===
            "pure_orange": {
                "emotions": {
                    "Happiness": 0.880, "Excitement": 0.820, "Confidence": 0.750,
                    "Pleasure": 0.680, "Anticipation": 0.590
                },
                "hue_range": [(15, 35)], "saturation_min": 0.5, "value_min": 0.5
            },
            "dark_orange": {
                "emotions": {
                    "Confidence": 0.680, "Engagement": 0.620, "Sympathy": 0.580,
                    "Sensitivity": 0.520, "Annoyance": 0.450
                },
                "hue_range": [(15, 35)], "saturation_min": 0.3, "value_max": 0.5
            },
            "light_orange": {
                "emotions": {
                    "Happiness": 0.780, "Pleasure": 0.720, "Affection": 0.650,
                    "Sympathy": 0.580, "Sensitivity": 0.520
                },
                "hue_range": [(15, 35)], "saturation_range": (0.2, 0.5), "value_min": 0.7
            },

            # === 노란색 계열 ===
            "pure_yellow": {
                "emotions": {
                    "Happiness": 0.920, "Surprise": 0.780, "Excitement": 0.750,
                    "Anticipation": 0.680, "Confidence": 0.620
                },
                "hue_range": [(35, 65)], "saturation_min": 0.6, "value_min": 0.7
            },
            "dark_yellow": {
                "emotions": {
                    "Disquietment": 0.620, "Aversion": 0.580, "Annoyance": 0.540,
                    "Sensitivity": 0.500, "Pain": 0.450
                },
                "hue_range": [(35, 65)], "saturation_min": 0.3, "value_max": 0.6
            },
            "light_yellow": {
                "emotions": {
                    "Happiness": 0.720, "Pleasure": 0.680, "Sympathy": 0.620,
                    "Affection": 0.580, "Sensitivity": 0.520
                },
                "hue_range": [(35, 65)], "saturation_range": (0.1, 0.4), "value_min": 0.8
            },

            # === 초록색 계열 ===
            "pure_green": {
                "emotions": {
                    "Pleasure": 0.820, "Sympathy": 0.780, "Affection": 0.720,
                    "Engagement": 0.680, "Sensitivity": 0.580
                },
                "hue_range": [(65, 140)], "saturation_min": 0.4, "value_min": 0.3
            },
            "dark_green": {
                "emotions": {
                    "Engagement": 0.680, "Sympathy": 0.620, "Sensitivity": 0.580,
                    "Disquietment": 0.520, "Aversion": 0.480
                },
                "hue_range": [(65, 140)], "saturation_min": 0.3, "value_max": 0.4
            },
            "light_green": {
                "emotions": {
                    "Pleasure": 0.750, "Sympathy": 0.690, "Affection": 0.640,
                    "Sensitivity": 0.590, "Happiness": 0.520
                },
                "hue_range": [(65, 140)], "saturation_range": (0.2, 0.5), "value_min": 0.6
            },

            # === 파란색 계열 ===
            "pure_blue": {
                "emotions": {
                    "Happiness": 0.720, "Pleasure": 0.680, "Confidence": 0.620,
                    "Engagement": 0.580, "Sensitivity": 0.480
                },
                "hue_range": [(140, 240)], "saturation_min": 0.5, "value_min": 0.4
            },
            "dark_blue": {
                "emotions": {
                    "Confidence": 0.680, "Engagement": 0.620, "Yearning": 0.580,
                    "Sensitivity": 0.520, "Disquietment": 0.480
                },
                "hue_range": [(140, 240)], "saturation_min": 0.3, "value_max": 0.4
            },
            "light_blue": {
                "emotions": {
                    "Happiness": 0.820, "Pleasure": 0.780, "Confidence": 0.720,
                    "Affection": 0.650, "Sensitivity": 0.420
                },
                "hue_range": [(140, 240)], "saturation_range": (0.2, 0.6), "value_min": 0.6
            },

            # === 보라색 계열 ===
            "pure_purple": {
                "emotions": {
                    "Yearning": 0.820, "Sensitivity": 0.750, "Embarrassment": 0.680,
                    "Surprise": 0.620, "Anticipation": 0.540
                },
                "hue_range": [(240, 300)], "saturation_min": 0.5, "value_min": 0.4
            },
            "dark_purple": {
                "emotions": {
                    "Fear": 0.680, "Disquietment": 0.640, "Yearning": 0.600,
                    "Pain": 0.560, "Aversion": 0.520
                },
                "hue_range": [(240, 300)], "saturation_min": 0.3, "value_max": 0.4
            },
            "light_purple": {
                "emotions": {
                    "Sensitivity": 0.720, "Affection": 0.680, "Sympathy": 0.640,
                    "Pleasure": 0.580, "Embarrassment": 0.520
                },
                "hue_range": [(240, 300)], "saturation_range": (0.2, 0.5), "value_min": 0.6
            },

            # === 분홍색 계열 ===
            "pure_pink": {
                "emotions": {
                    "Affection": 0.880, "Happiness": 0.780, "Embarrassment": 0.680,
                    "Pleasure": 0.640, "Sympathy": 0.580
                },
                "hue_range": [(300, 340)], "saturation_min": 0.4, "value_min": 0.5
            },
            "dark_pink": {
                "emotions": {
                    "Affection": 0.720, "Sympathy": 0.680, "Sensitivity": 0.620,
                    "Embarrassment": 0.580, "Yearning": 0.520
                },
                "hue_range": [(300, 340)], "saturation_min": 0.3, "value_max": 0.6
            },
            "light_pink": {
                "emotions": {
                    "Affection": 0.820, "Happiness": 0.750, "Pleasure": 0.680,
                    "Sympathy": 0.620, "Sensitivity": 0.580
                },
                "hue_range": [(300, 340)], "saturation_range": (0.1, 0.4), "value_min": 0.7
            },

            # === 갈색 계열 ===
            "brown": {
                "emotions": {
                    "Sympathy": 0.680, "Sensitivity": 0.640, "Engagement": 0.580,
                    "Disquietment": 0.520, "Aversion": 0.480
                },
                "hue_range": [(10, 35)], "saturation_range": (0.2, 0.6), "value_range": (0.2, 0.7)
            },

            # === 무채색 계열 ===
            "white": {
                "emotions": {
                    "Happiness": 0.750, "Surprise": 0.680, "Pleasure": 0.620,
                    "Confidence": 0.580, "Sensitivity": 0.520
                },
                "hue_range": [(0, 360)], "saturation_max": 0.1, "value_min": 0.9
            },
            "cream": {
                "emotions": {
                    "Sympathy": 0.720, "Sensitivity": 0.680, "Affection": 0.620,
                    "Pleasure": 0.580, "Happiness": 0.520
                },
                "hue_range": [(0, 360)], "saturation_range": (0.05, 0.25), "value_min": 0.85
            },
            "gray": {
                "emotions": {
                    "Disquietment": 0.720, "Pain": 0.650, "Aversion": 0.580,
                    "Sensitivity": 0.520, "Sympathy": 0.440
                },
                "hue_range": [(0, 360)], "saturation_max": 0.2, "value_range": (0.3, 0.85)
            },
            "black": {
                "emotions": {
                    "Fear": 0.850, "Disquietment": 0.780, "Pain": 0.720,
                    "Aversion": 0.680, "Annoyance": 0.580
                },
                "hue_range": [(0, 360)], "saturation_max": 0.3, "value_max": 0.3
            }

        }

    def extract_dominant_colors(self, image_path: str, n_colors: int = 5,
                                resize_width: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        [이력서 전략: CPU 전용 연산]
        - K-means 클러스터링을 GPU가 아닌 CPU에서 수행하여 VRAM 점유 0MB 유지
        - 분석 속도를 위해 resize_width를 100px로 최적화
        """
        import logging
        logger = logging.getLogger(__name__)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        new_height = int(h * (resize_width / w))
        image = cv2.resize(image, (resize_width, new_height), interpolation=cv2.INTER_AREA)

        pixels = image.reshape(-1, 3)

        logger.info(f"[CPU Task] K-Means 분석 시작 (Pixels: {pixels.shape[0]})")
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        percentages = counts / len(kmeans.labels_)

        sorted_indices = np.argsort(percentages)[::-1]
        return colors[sorted_indices], percentages[sorted_indices]

    def rgb_to_hsv(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        """RGB를 HSV로 변환"""
        r, g, b = rgb / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s, v  # H를 0-360도로 변환

    def classify_color(self, rgb: np.ndarray) -> List[str]:
        """RGB 색상을 색상 카테고리로 분류"""
        h, s, v = self.rgb_to_hsv(rgb)
        matched_categories = []

        for category, config in self.color_emotion_mapping.items():
            # 색조(Hue) 체크
            hue_match = False
            for hue_min, hue_max in config["hue_range"]:
                if hue_min <= h <= hue_max:
                    hue_match = True
                    break

            if not hue_match:
                continue

            # 채도(Saturation) 체크
            if "saturation_min" in config and s < config["saturation_min"]:
                continue
            if "saturation_max" in config and s > config["saturation_max"]:
                continue
            if "saturation_range" in config:
                s_min, s_max = config["saturation_range"]
                if not (s_min <= s <= s_max):
                    continue

            # 명도(Value) 체크
            if "value_min" in config and v < config["value_min"]:
                continue
            if "value_max" in config and v > config["value_max"]:
                continue
            if "value_range" in config:
                v_min, v_max = config["value_range"]
                if not (v_min <= v <= v_max):
                    continue

            matched_categories.append(category)

        return matched_categories

    def predict_emotions(self, image_path: str, n_colors: int = 5) -> Dict:

        # 주요 색상 추출 (CPU 연산)
        colors, percentages = self.extract_dominant_colors(image_path, n_colors)

        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        color_analysis = []

        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            h, s, v = self.rgb_to_hsv(color)
            categories = self.classify_color(color)

            color_info = {
                "rank": i + 1,
                "rgb": color.tolist(),
                "percentage": round(percentage * 100, 2),
                "categories": categories
            }

            # 감정 점수 누적
            for category in categories:
                config = self.color_emotion_mapping[category]
                for emotion, emotion_percentage in config["emotions"].items():
                    # 가중치 계산: 색상 비율 * 감정 강도
                    score = percentage * emotion_percentage
                    emotion_scores[emotion] += score

            color_analysis.append(color_info)

        # 결과 정리 및 상위 감정 추출
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            percentage_scores = {e: (s / total_score) * 100 for e, s in emotion_scores.items()}
        else:
            percentage_scores = {e: 0.0 for e in self.emotions}

        sorted_emotions = sorted(percentage_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "method": "cpu_color_analysis",
            "image_path": str(image_path),
            "dominant_colors": color_analysis,
            "top_emotions": [(e, round(p, 3)) for e, p in sorted_emotions[:5]],
            "predicted_emotions": [e for e, p in sorted_emotions if p > 5.0]
        }

# 테스트용 메인 함수 (단독 실행시에만 동작)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Color-based Emotion Inference Module")
    parser.add_argument('--image', required=True, help='Image file path')
    parser.add_argument('--colors', type=int, default=5, help='Number of colors to extract')

    args = parser.parse_args()

    # 색상 감정 분석기 테스트
    analyzer = ColorEmotionInference()
    result = analyzer.predict_emotions(args.image, n_colors=args.colors)

    print("\n" + "="*60)
    print("COLOR EMOTION ANALYSIS (MODULE TEST)")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Colors extracted: {len(result['dominant_colors'])}")

    print("\nDominant Colors:")
    for color_info in result['dominant_colors']:
        rgb = color_info['rgb']
        pct = color_info['percentage']
        categories = color_info['categories']
        print(f"  #{color_info['rank']}: RGB{rgb} ({pct:.1f}%) - {categories}")

    print(f"\nTop 5 Emotions:")
    for emotion, percentage in result['top_emotions']:
        print(f"  {emotion:15s}: {percentage:.3f}%")

    print(f"\nPredicted Emotions (>5%): {result['predicted_emotions']}")