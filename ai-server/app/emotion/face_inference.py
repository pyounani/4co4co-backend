import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from emotic import Emotic
    from inference import infer
    from yolo_utils import (
        prepare_yolo as prepare_yolo_v3,
        rescale_boxes as rescale_boxes_v3,
        non_max_suppression as non_max_suppression_v3,
    )
    EMOTIC_AVAILABLE = True
except ImportError:
    print("EMOTIC 모듈을 찾을 수 없습니다. emotic, inference, yolo_utils 모듈이 필요합니다.")
    EMOTIC_AVAILABLE = False


class EmoticdAdaptive(nn.Module):
    """적응형 EMOTIC 모델 - 입력 차원을 자동으로 조정"""
    def __init__(self, context_dim=None, body_dim=None):
        super(EmoticdAdaptive, self).__init__()
        self.context_dim = context_dim
        self.body_dim = body_dim
        self.layers_initialized = False
        self.fc1 = None; self.bn1 = None; self.d1 = None
        self.fc_cat = None; self.fc_cont = None
        self.relu = nn.ReLU()

    def _initialize_layers(self, context_features, body_features):
        ctx = context_features.view(context_features.size(0), -1)
        bod = body_features.view(body_features.size(0), -1)
        self.context_dim = ctx.size(1)
        self.body_dim = bod.size(1)
        total_dim = self.context_dim + self.body_dim
        print(f"Adaptive EMOTIC: Context dim={self.context_dim}, Body dim={self.body_dim}, Total={total_dim}")
        self.fc1    = nn.Linear(total_dim, 256)
        self.bn1    = nn.BatchNorm1d(256)
        self.d1     = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont= nn.Linear(256, 3)
        self.layers_initialized = True

    def forward(self, x_context, x_body):
        if not self.layers_initialized:
            self._initialize_layers(x_context, x_body)
        ctx = x_context.view(x_context.size(0), -1)
        bod = x_body.view(x_body.size(0), -1)
        fuse = torch.cat((ctx, bod), 1)
        h = self.fc1(fuse)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.d1(h)
        return self.fc_cat(h), self.fc_cont(h)


class FaceEmotionAnalyzer:
    """EMOTIC 모델을 사용한 얼굴/표정 감정 분석기"""

    def __init__(self, experiment_path: str = ".",
                 model_dir: str = "emotic",     
                 result_dir: str = "results",
                 gpu: int = 0,
                 yolo_model_dir: str = "yolo"):     
        if not EMOTIC_AVAILABLE:
            raise ImportError("EMOTIC 모듈이 설치되지 않았습니다.")

        self.experiment_path = experiment_path
        self.emotic_model_path = os.path.join(experiment_path, model_dir)
        self.yolo_model_path = os.path.join(experiment_path, yolo_model_dir)
        self.result_path = os.path.join(experiment_path, result_dir)
        self.gpu = gpu

        self._check_paths()

        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        print(f"EMOTIC 감정 분석기 초기화 완료 (Device: {self.device})")
        print("EMOTIC dir:", os.path.abspath(self.emotic_model_path))
        print("YOLOv3 dir:", os.path.abspath(self.yolo_model_path))

        self.emotions = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
                         'Confidence', 'Disapproval', 'Disconnection', 'Disquietment',
                         'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
                         'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
                         'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
                         'Sympathy', 'Yearning']
        self.cat2ind = {e: i for i, e in enumerate(self.emotions)}
        self.ind2cat = {i: e for i, e in enumerate(self.emotions)}
        self.vad = ['Valence', 'Arousal', 'Dominance']
        self.ind2vad = {i: v for i, v in enumerate(self.vad)}

        self.context_norm = [[0.4690646, 0.4407227, 0.40508908],
                             [0.2514227, 0.24312855, 0.24266963]]
        self.body_norm = [[0.43832874, 0.3964344, 0.3706214],
                          [0.24784276, 0.23621225, 0.2323653]]

        self.models_loaded = False
        self.yolo = None
        self.models = None
        self.thresholds = None

    def _check_paths(self):
        if not os.path.exists(self.emotic_model_path):
            raise ValueError(f'EMOTIC model path {self.emotic_model_path} does not exist.')
        if not os.path.exists(self.yolo_model_path):
            raise ValueError(f'YOLOv3 path {self.yolo_model_path} does not exist.')
        os.makedirs(self.result_path, exist_ok=True)

    def diagnose_model_dimensions(self):
        print(" EMOTIC 모델 차원 진단 시작...")
        for model_file in ['model_context1.pth', 'model_body1.pth', 'model_emotic1.pth']:
            file_path = os.path.join(self.emotic_model_path, model_file)
            if os.path.exists(file_path):
                try:
                    model_data = torch.load(file_path, map_location='cpu')
                    print(f"\n {model_file}:")
                    if hasattr(model_data, 'state_dict'):
                        state_dict = model_data.state_dict()
                    elif isinstance(model_data, dict):
                        state_dict = model_data
                    else:
                        print(f"  타입: {type(model_data)}")
                        continue
                    for key, tensor in state_dict.items():
                        try:
                            print(f"  {key}: {tuple(tensor.shape)}")
                        except Exception:
                            print(f"  {key}: <non-tensor>")
                except Exception as e:
                    print(f"   {model_file} 분석 실패: {e}")
            else:
                print(f"   {model_file} 파일이 존재하지 않습니다.")

    def _load_models(self):
        if self.models_loaded:
            return
        print("전처리 모델들을 CPU 메모리에 로딩 중 (VRAM 점유 0MB 유지)...")

        self.yolo = prepare_yolo_v3(self.yolo_model_path).to('cpu').eval()

        ctx_p = os.path.join(self.emotic_model_path, 'model_context1.pth')
        body_p = os.path.join(self.emotic_model_path, 'model_body1.pth')
        head_p = os.path.join(self.emotic_model_path, 'model_emotic1.pth')

        model_context = torch.load(ctx_p, map_location='cpu')
        model_body = torch.load(body_p, map_location='cpu')
        emotic_head = torch.load(head_p, map_location='cpu')

        for m in (model_context, model_body, emotic_head):
            if hasattr(m, 'eval'): m.eval()

        self.models = [model_context, model_body, emotic_head]
        self.models_loaded = True
        print("전처리 모델 로드 완료 (CPU 대기 모드)")

    def get_bbox(self, image_context: np.ndarray, yolo_image_size: int = 416,
                 conf_thresh: float = 0.8, nms_thresh: float = 0.4) -> np.ndarray:
        """YOLOv3로 사람 박스 검출"""
        import gc
        try:
            self.yolo.to(self.device)

            test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            image_yolo = test_transform(cv2.resize(image_context, (yolo_image_size, yolo_image_size))).unsqueeze(0).to(self.device)

            with torch.no_grad():
                detections = self.yolo(image_yolo)
                nms_det = non_max_suppression_v3(detections, conf_thresh, nms_thresh)[0]
                if nms_det is None or len(nms_det) == 0:
                    return np.empty((0, 4), dtype=int)
                det = rescale_boxes_v3(nms_det, yolo_image_size, (image_context.shape[:2]))

            if torch.is_tensor(det):
                det = det.detach().cpu().numpy()

            bboxes = []
            for row in det:
                x1, y1, x2, y2, _, _, cls_pred = row.tolist()
                if int(cls_pred) == 0:  # person class
                    x1 = int(min(image_context.shape[1], max(0,   x1)))
                    x2 = int(min(image_context.shape[1], max(x1,  x2)))
                    y1 = int(min(image_context.shape[0], max(0,   y1)))
                    y2 = int(min(image_context.shape[0], max(y1,  y2)))
                    bboxes.append([x1, y1, x2, y2])
            return np.asarray(bboxes, dtype=int)

        finally:
            self.yolo.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[YOLO] VRAM 반환 완료")

    def analyze_emotions(self, image_path: str, person_detections: Optional[List[Dict]] = None) -> Dict:
        import gc
        import os
        import cv2
        import numpy as np
        import torch

        if not os.path.exists(image_path):
            return {"error": f"Image file {image_path} does not exist"}

        try:
            self._load_models()
        except Exception as e:
            return {"error": f"Failed to load models: {str(e)}"}

        try:
            image_context = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image_context is None:
                return {"error": f"Failed to load image: {image_path}"}
        except Exception as e:
            return {"error": f"Error loading image: {str(e)}"}

        if person_detections is None:
            try:
                bbox_yolo = self.get_bbox(image_context)
                person_count = len(bbox_yolo)
            except Exception as e:
                return {"error": f"Person detection failed: {str(e)}"}
        else:
            bbox_yolo = []
            for detection in person_detections:
                if 'bbox' in detection and len(detection['bbox']) == 4:
                    bbox_yolo.append(detection['bbox'])
            bbox_yolo = np.asarray(bbox_yolo, dtype=int) if bbox_yolo else np.empty((0, 4), dtype=int)
            person_count = len(bbox_yolo)

        if person_count == 0:
            return {
                "method": "emotic_face_emotion_dynamic",
                "person_count": 0,
                "emotion_percentages": {emotion: 0.0 for emotion in self.emotions},
                "top_emotions": [],
                "predicted_emotions": [],
                "message": "No person detected in the image"
            }

        all_emotion_scores, all_vad_scores = [], []
        successful_analyses = 0

        try:
            if self.device.type == 'cuda':
                for m in self.models:
                    m.to(self.device)

            for bbox_idx, pred_bbox in enumerate(bbox_yolo):
                try:
                    pred_bbox = np.asarray(pred_bbox, dtype=int)
                    result = infer(self.context_norm, self.body_norm, self.ind2cat, self.ind2vad,
                                   self.device, self.thresholds, self.models,
                                   image_context=image_context, bbox=pred_bbox, to_print=False)

                    if len(result) >= 4:
                        pred_cat, pred_cont, emotion_dict, sorted_emotions = result
                        all_emotion_scores.append(emotion_dict)
                        all_vad_scores.append(pred_cont)
                        successful_analyses += 1
                        print(f"[EMOTIC] Person {bbox_idx + 1} 분석 완료")
                except Exception as e:
                    print(f"Error analyzing person {bbox_idx + 1}: {str(e)}")
                    continue

            if successful_analyses == 0:
                return {
                    "method": "emotic_face_emotion_dynamic",
                    "person_count": person_count,
                    "message": "Face analysis failed for all persons"
                }

            averaged_emotions = {}
            for emotion in self.emotions:
                scores = [emotion_scores.get(emotion, 0.0) for emotion_scores in all_emotion_scores]
                averaged_emotions[emotion] = sum(scores) / len(scores) * 100.0

            avg_vad = np.mean(all_vad_scores, axis=0) if all_vad_scores else [0.0, 0.0, 0.0]

            sorted_emotions_final = sorted(averaged_emotions.items(), key=lambda x: x[1], reverse=True)
            predicted_emotions = [emotion for emotion, score in sorted_emotions_final if score > 5.0]

            return {
                "method": "emotic_face_emotion_dynamic",
                "person_count": person_count,
                "successful_analyses": successful_analyses,
                "emotion_percentages": {e: round(p, 3) for e, p in averaged_emotions.items()},
                "top_emotions": [(e, round(p, 3)) for e, p in sorted_emotions_final[:5]],
                "predicted_emotions": predicted_emotions,
                "vad_dimensions": {
                    "valence": float(avg_vad[0]),
                    "arousal": float(avg_vad[1]),
                    "dominance": float(avg_vad[2])
                }
            }

        except Exception as pipeline_error:
            print(f"추론 파이프라인 에러: {pipeline_error}")
            return {"error": str(pipeline_error)}

        finally:
            if self.models_loaded:
                for m in self.models:
                    m.to('cpu')

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[EMOTIC] VRAM Purge 완료. 상시 적재 모델(MusicGen) 가용 공간 확보.")

    def analyze_single_image(self, image_path: str, conf_thresh: float = 0.8,
                             nms_thresh: float = 0.4, save_result: bool = False) -> Dict:
        """단일 이미지 분석 + 시각화(선택)"""
        try:
            self._load_models()
        except Exception as e:
            return {"error": f"Failed to load models: {str(e)}"}

        if not os.path.exists(image_path):
            return {"error": f"Image file {image_path} does not exist"}

        image_context = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if image_context is None:
            return {"error": f"Failed to load image: {image_path}"}

        bboxes = self.get_bbox(image_context, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        if len(bboxes) == 0:
            return {"image_path": image_path, "person_count": 0, "message": "No person detected"}

        analysis_results = []
        for i, bbox in enumerate(bboxes):
            bbox = np.asarray(bbox, dtype=int)
            try:
                pred_cat, pred_cont, emotion_dict, sorted_emotions = infer(
                    self.context_norm, self.body_norm, self.ind2cat, self.ind2vad,
                    self.device, self.thresholds, self.models,
                    image_context=image_context, bbox=bbox, to_print=False
                )
                analysis_results.append({
                    "person_id": i + 1,
                    "bbox": bbox.tolist(),
                    "predicted_emotions": pred_cat,
                    "vad_scores": [float(x) for x in pred_cont],
                    "top_emotions": [(e, float(s*100.0)) for e, s in sorted_emotions[:5]]
                })

                if save_result:
                    vad_text = f"VAD {pred_cont[0]:.1f} {pred_cont[1]:.1f} {pred_cont[2]:.1f}"
                    image_context = cv2.rectangle(image_context, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
                    cv2.putText(image_context, vad_text, (bbox[0], bbox[1]-5),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    for j, emotion in enumerate(pred_cat[:3]):
                        cv2.putText(image_context, emotion, (bbox[0], bbox[1] + (j+1)*12),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            except Exception as e:
                analysis_results.append({"person_id": i + 1, "bbox": bbox.tolist(), "error": str(e)})

        output_path = None
        if save_result and len(analysis_results) > 0:
            os.makedirs(self.result_path, exist_ok=True)
            fname = f"emotic_result_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
            output_path = os.path.join(self.result_path, fname)
            cv2.imwrite(output_path, cv2.cvtColor(image_context, cv2.COLOR_RGB2BGR))

        return {
            "image_path": image_path,
            "person_count": len(bboxes),
            "analysis_results": analysis_results,
            "output_path": output_path
        }


# 단독 실행을 위한 CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EMOTIC Face Emotion Analyzer (YOLOv3)")
    parser.add_argument('--image', type=str, help='Single image path for analysis')
    parser.add_argument('--experiment_path', type=str, default='.', help='Experiment path')
    parser.add_argument('--model_dir', type=str, default='model/emotic', help='EMOTIC model directory')
    parser.add_argument('--yolo_model_dir', type=str, default='model/yolo', help='YOLOv3 model directory')
    parser.add_argument('--result_dir', type=str, default='results', help='Result directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--conf_thresh', type=float, default=0.8, help='YOLO confidence threshold')
    parser.add_argument('--save_result', action='store_true', help='Save result image')
    parser.add_argument('--diagnose', action='store_true', help='Diagnose model dimensions')
    args = parser.parse_args()

    analyzer = FaceEmotionAnalyzer(
        experiment_path=args.experiment_path,
        model_dir=args.model_dir,           # ./model/emotic
        yolo_model_dir=args.yolo_model_dir, # ./model/yolo
        result_dir=args.result_dir,
        gpu=args.gpu
    )

    if args.diagnose:
        analyzer.diagnose_model_dimensions()
    elif args.image:
        result = analyzer.analyze_single_image(
            image_path=args.image,
            conf_thresh=args.conf_thresh,
            save_result=args.save_result
        )
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\n Final Results:")
            print(f"Image: {result['image_path']}")
            print(f"Persons detected: {result['person_count']}")
            ok = len([r for r in result['analysis_results'] if 'error' not in r])
            print(f"Successful analyses: {ok}")
            if result.get("output_path"):
                print(f"Saved: {result['output_path']}")
    else:
        parser.print_help()
