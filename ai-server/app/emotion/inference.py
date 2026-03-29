import os

import cv2
import numpy as np
import torch
from torchvision import transforms


def process_images(context_norm, body_norm, image_context_path=None, image_context=None, image_body=None, bbox=None):
  if image_context is None and image_context_path is None:
    raise ValueError('both image_context and image_context_path cannot be none. Please specify one of the two.')
  if image_body is None and bbox is None:
    raise ValueError('both body image and bounding box cannot be none. Please specify one of the two')

  if image_context_path is not None:
    image_context =  cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)

  if bbox is not None:
    image_body = image_context[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()

  image_context = cv2.resize(image_context, (224,224))
  image_body = cv2.resize(image_body, (128,128))

  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  context_norm = transforms.Normalize(context_norm[0], context_norm[1])
  body_norm = transforms.Normalize(body_norm[0], body_norm[1])

  image_context = context_norm(test_transform(image_context)).unsqueeze(0)
  image_body = body_norm(test_transform(image_body)).unsqueeze(0)

  return image_context, image_body


def infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models,
          image_context_path=None, image_context=None, image_body=None, bbox=None, to_print=True):

  import gc
  import torch

  image_context, image_body = process_images(
    context_norm, body_norm,
    image_context_path=image_context_path,
    image_context=image_context,
    image_body=image_body,
    bbox=bbox
  )

  model_context, model_body, emotic_model = models

  with torch.no_grad():
    image_context = image_context.to(device)
    image_body = image_body.to(device)

    pred_context = model_context(image_context)
    del image_context

    pred_body = model_body(image_body)
    del image_body

    p_cat, p_cont = emotic_model(pred_context, pred_body)

    del pred_context, pred_body

    pred_cat = p_cat.squeeze(0).float().cpu()
    pred_cont = p_cont.squeeze(0).cpu().numpy()

    del p_cat, p_cont

    if pred_cat.min() < 0.0 or pred_cat.max() > 1.0:
      probs = torch.sigmoid(pred_cat)
    else:
      probs = pred_cat

  if thresholds.is_cuda:
    thresholds = thresholds.cpu()

  bool_cat_pred = torch.gt(probs, thresholds)

  cat_emotions = []
  selected_with_probs = []
  for i in range(len(bool_cat_pred)):
    if bool_cat_pred[i]:
      emotion = ind2cat[i]
      p = probs[i].item()
      cat_emotions.append(emotion)
      selected_with_probs.append((emotion, p))

  final_vad = 10 * pred_cont

  if to_print:
    print(f'\n[Inference] Categorical Emotions: {cat_emotions}')
    print(f'[Inference] VAD Scores: {final_vad}')

  gc.collect()
  if torch.cuda.is_available(): torch.cuda.empty_cache()

  probs_dict = {ind2cat[i]: float(probs[i].item()) for i in range(len(probs))}

  return cat_emotions, final_vad, probs_dict, selected_with_probs


def inference_emotic(images_list, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad, args):

  import gc
  import torch
  import os
  import numpy as np

  with open(images_list, 'r') as f:
    lines = f.readlines()

  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  print(f"[VRAM Strategy] 초기 상태: 모델을 CPU 메모리에 로드합니다.")

  model_context = torch.load(os.path.join(model_path, 'model_context1.pth'), map_location='cpu')
  model_body = torch.load(os.path.join(model_path, 'model_body1.pth'), map_location='cpu')
  emotic_model = torch.load(os.path.join(model_path, 'model_emotic1.pth'), map_location='cpu')

  thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy')))

  models = [model_context, model_body, emotic_model]

  result_file = os.path.join(result_path, 'inference_list.txt')
  result_file_probs = os.path.join(result_path, 'inference_list_with_probs.txt')
  for fp in [result_file, result_file_probs]:
    with open(fp, 'w') as f: pass

  for idx, line in enumerate(lines):
    try:
      if device.type == 'cuda':
        for m in models:
          m.to(device).eval()

      parts = line.split('\n')[0].split(' ')
      image_context_path = parts[0]
      bbox = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]

      pred_cat, pred_cont, probs_dict, selected_with_probs = infer(
        context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models,
        image_context_path=image_context_path, bbox=bbox, to_print=False)

      write_line = [image_context_path] + pred_cat + [f'{c:.4f}' for c in pred_cont]
      with open(result_file, 'a') as f:
        f.write(' '.join(write_line) + '\n')

      selected_str = ', '.join([f'{e}:{p:.4f}' for e, p in selected_with_probs]) if selected_with_probs else '(none)'
      line_probs = f"{image_context_path} | SELECTED: {selected_str} | VAD: {', '.join([f'{v:.4f}' for v in pred_cont])}"
      with open(result_file_probs, 'a') as f:
        f.write(line_probs + '\n')

      print(f"[Batch] {idx + 1}/{len(lines)} 이미지 처리 완료")

    except Exception as e:
      print(f"[Error] {idx + 1}번째 이미지 처리 중 오류: {e}")
      continue

    finally:
      if device.type == 'cuda':
        for m in models:
          m.to('cpu')

      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  print(f"[VRAM Strategy] 배치 추론 종료. 모든 모델이 CPU로 반환되었습니다.")
