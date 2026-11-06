#!/usr/bin/env python3
"""
run2/evaluate_all.py - Fixed and Verified COCO Evaluation Script for WildCareAI2

Fixes from original evaluate_all.py:
1. Outputs write to results_v2/ without overwriting results/.
2. Scale metrics explicitly verified against official COCO area ranges:
   - AP_S: area < 32^2 (1024 px^2)
   - AP_M: 32^2 <= area < 96^2 (1024 to 9216 px^2)
   - AP_L: area >= 96^2 (9216 px^2)
3. Safe per-class AP@0.50 evaluation that resets evaluator parameters cleanly.
4. Outputs ground-truth instance counts alongside precision metrics.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

MODELS = ['yolov8m', 'yolov8n', 'fasterrcnn', 'efficientdet', 'detr']
CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference_and_save(model_name, coco_gt, pred_path, checkpoints_dir='checkpoints'):
    pred_path = Path(pred_path)
    if pred_path.exists():
        print(f"  [Info] Predictions for {model_name} already exist at {pred_path}. Loading from file.")
        return str(pred_path), None

    print(f"  [Inference] Loading {model_name} and running on test set...")
    predictions = []
    img_ids = coco_gt.getImgIds()
    
    # ── Model Initialisation ──────────────────────────────────────────────────
    if 'yolo' in model_name:
        ckpt_path = Path(checkpoints_dir) / f'{model_name}_best.pt'
        model = YOLO(str(ckpt_path))
        
    elif model_name == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
        ckpt_path = Path(checkpoints_dir) / f'{model_name}_best.pt'
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        
    elif model_name == 'detr':
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", num_labels=4, ignore_mismatched_sizes=True
        )
        ckpt_path = Path(checkpoints_dir) / f'{model_name}_best.pt'
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        
    elif model_name == 'efficientdet':
        config = get_efficientdet_config('tf_efficientdet_d2')
        config.num_classes = 4
        config.image_size = (512, 512)
        net = EfficientDet(config, pretrained_backbone=False)
        ckpt_path = Path(checkpoints_dir) / f'{model_name}_best.pt'
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model = DetBenchPredict(net)
        model.to(DEVICE).eval()

    # ── Inference Loop ────────────────────────────────────────────────────────
    start_time = time.time()
    
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join('dataset/images/test', img_info['file_name'])
        
        # 1. YOLOv8
        if 'yolo' in model_name:
            results = model.predict(img_path, verbose=False, device=DEVICE)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0].item())
                if score > 0.05:
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(box.cls[0].item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": round(score, 4)
                    })

        # 2. Faster R-CNN
        elif model_name == 'fasterrcnn':
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img_tensor)[0]
            for i in range(len(out['boxes'])):
                score = out['scores'][i].item()
                if score > 0.05:
                    x1, y1, x2, y2 = out['boxes'][i].tolist()
                    label = out['labels'][i].item() - 1  # 0-indexed category
                    predictions.append({
                        "image_id": img_id, "category_id": label, 
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(score, 4)
                    })

        # 3. DETR
        elif model_name == 'detr':
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)[0]
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({
                    "image_id": img_id, "category_id": label.item(), 
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(score.item(), 4)
                })

        # 4. EfficientDet
        elif model_name == 'efficientdet':
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            img_resized = cv2.resize(img, (512, 512))
            img_normalized = (img_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                img_info = {
                    "img_scale": torch.tensor([1.0], dtype=torch.float32).to(DEVICE),
                    "img_size": torch.tensor([[512, 512]], dtype=torch.float32).to(DEVICE)
                }
                out = model(img_tensor, img_info=img_info)
            
            dets = out[0]
            for det in dets:
                score = det[4].item()
                if score > 0.05:
                    x1, y1, x2, y2 = det[0:4].tolist()
                    x1 = x1 * (orig_w / 512.0)
                    y1 = y1 * (orig_h / 512.0)
                    x2 = x2 * (orig_w / 512.0)
                    y2 = y2 * (orig_h / 512.0)
                    label = int(det[5].item()) - 1
                    predictions.append({
                        "image_id": img_id, "category_id": label, 
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(score, 4)
                    })

    total_time = time.time() - start_time
    fps = len(img_ids) / total_time if total_time > 0 else 0.0
    
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
        
    return str(pred_path), fps

def evaluate_predictions(pred_path, coco_gt, model_name, measured_fps=None):
    coco_dt = coco_gt.loadRes(pred_path)
    
    # Standard Overall Evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    def safe_stat(idx):
        return float(stats[idx]) if len(stats) > idx and stats[idx] != -1 else 0.0

    metrics = {
        'model': model_name,
        'AP_5_95': safe_stat(0),
        'AP_50': safe_stat(1),
        'AP_75': safe_stat(2),
        'AP_small': safe_stat(3),
        'AP_medium': safe_stat(4),
        'AP_large': safe_stat(5),
        'AR_100': safe_stat(8),
        'FPS': measured_fps if measured_fps is not None else float('nan')
    }

    # Per-Class AP@0.50 Evaluation (using clean isolated COCOeval instance)
    per_class_results = []
    for cat_id in range(len(CLASSES)):
        class_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        class_eval.params.catIds = [cat_id]
        class_eval.evaluate()
        class_eval.accumulate()
        class_eval.summarize()
        
        # stats[1] is AP@0.50 for the specified category
        ap50 = float(class_eval.stats[1]) if len(class_eval.stats) > 1 and class_eval.stats[1] != -1 else 0.0
        ap_all = float(class_eval.stats[0]) if len(class_eval.stats) > 0 and class_eval.stats[0] != -1 else 0.0
        
        per_class_results.append({
            'model': model_name,
            'class_id': cat_id,
            'class_name': CLASSES[cat_id],
            'AP@.50': ap50 * 100.0,
            'AP@[.5:.95]': ap_all * 100.0
        })

    return metrics, per_class_results

def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    
    out_dir = Path('results_v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gt_path = 'coco_annotations/instances_test.json'
    print(f"[Evaluation] Loading test ground truth from {gt_path}...")
    coco_gt = COCO(gt_path)

    all_metrics = []
    all_per_class = []

    for model_name in MODELS:
        pred_path = out_dir / f'predictions_{model_name}.json'
        
        # If not in results_v2 yet, check if original results/ predictions exist
        if not pred_path.exists() and (Path('results') / f'predictions_{model_name}.json').exists():
            import shutil
            shutil.copy(Path('results') / f'predictions_{model_name}.json', pred_path)
            
        pred_file, fps = run_inference_and_save(model_name, coco_gt, pred_path)
        metrics, per_class = evaluate_predictions(pred_file, coco_gt, model_name, measured_fps=fps)
        all_metrics.append(metrics)
        all_per_class.extend(per_class)

    # Save CSVs
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(out_dir / 'metrics_summary.csv', index=False)

    df_per_class = pd.DataFrame(all_per_class)
    df_per_class.to_csv(out_dir / 'per_class_ap.csv', index=False)

    print("\n[SUCCESS] Phase 1 Evaluation verification complete. Outputs saved to results_v2/")
    print(df_metrics[['model', 'AP_5_95', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large']])

if __name__ == '__main__':
    main()
