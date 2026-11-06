#!/usr/bin/env python3
"""
run2/benchmark_worker.py - Isolated Benchmark Worker for a Single Model Run
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['yolov8m', 'yolov8n', 'fasterrcnn', 'detr', 'efficientdet'])
    parser.add_argument('--mode', type=str, required=True, choices=['native', 'uniform640'])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    test_imgs = sorted(list(Path('dataset/images/test').glob('*.jpg')))
    raw_imgs = [(p.name, open(p, 'rb').read()) for p in test_imgs]

    model_name = args.model
    mode = args.mode

    if model_name in ['yolov8m', 'yolov8n']:
        from ultralytics import YOLO
        ckpt = 'checkpoints/yolov8m_best.pt' if model_name == 'yolov8m' else 'checkpoints/yolov8n_best.pt'
        model = YOLO(ckpt).to(DEVICE)
        
        # Warmup (25 images)
        for _, b in raw_imgs[:25]:
            arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            _ = model.predict(arr, imgsz=640, rect=False, verbose=False, device=0)
        torch.cuda.synchronize()

        mod_times = []
        e2e_times = []
        with torch.no_grad():
            for _, b in raw_imgs:
                torch.cuda.synchronize()
                t0_e2e = time.perf_counter()
                arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                res = model.predict(arr, imgsz=640, rect=False, verbose=False, device=0)
                _ = res[0].boxes.xyxy
                torch.cuda.synchronize()
                t1_e2e = time.perf_counter()

                mod_times.append(float(res[0].speed['inference']))
                e2e_times.append(float((t1_e2e - t0_e2e) * 1000.0))

    elif model_name == 'fasterrcnn':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import torchvision.transforms.functional as F

        min_sz = 640 if mode == 'uniform640' else 800
        max_sz = 640 if mode == 'uniform640' else 1333
        frcnn = fasterrcnn_resnet50_fpn(weights=None, min_size=min_sz, max_size=max_sz)
        frcnn.roi_heads.box_predictor = FastRCNNPredictor(frcnn.roi_heads.box_predictor.cls_score.in_features, num_classes=5)
        frcnn.load_state_dict(torch.load('checkpoints/fasterrcnn_best.pt', map_location=DEVICE))
        frcnn.to(DEVICE).eval()

        # Warmup (25 images)
        for _, b in raw_imgs[:25]:
            arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            t = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
            _ = frcnn(t)
        torch.cuda.synchronize()

        mod_times = []
        e2e_times = []
        with torch.no_grad():
            for _, b in raw_imgs:
                torch.cuda.synchronize()
                t0_e2e = time.perf_counter()
                arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(arr_rgb)
                tens = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
                
                torch.cuda.synchronize()
                t_m0 = time.perf_counter()
                out = frcnn(tens)[0]
                torch.cuda.synchronize()
                t_m1 = time.perf_counter()

                _ = out['boxes'].cpu()
                torch.cuda.synchronize()
                t1_e2e = time.perf_counter()

                mod_times.append(float((t_m1 - t_m0) * 1000.0))
                e2e_times.append(float((t1_e2e - t0_e2e) * 1000.0))

    elif model_name == 'detr':
        from transformers import DetrImageProcessor, DetrForObjectDetection

        sz_dict = {'shortest_edge': 640, 'longest_edge': 640} if mode == 'uniform640' else {'shortest_edge': 800, 'longest_edge': 1333}
        detr_proc = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', size=sz_dict)
        detr = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=4, ignore_mismatched_sizes=True)
        detr.load_state_dict(torch.load('checkpoints/detr_best.pt', map_location=DEVICE))
        detr.to(DEVICE).eval()

        # Warmup (25 images)
        for _, b in raw_imgs[:25]:
            arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            inp = detr_proc(images=pil, return_tensors='pt').to(DEVICE)
            _ = detr(**inp)
        torch.cuda.synchronize()

        mod_times = []
        e2e_times = []
        with torch.no_grad():
            for _, b in raw_imgs:
                torch.cuda.synchronize()
                t0_e2e = time.perf_counter()
                arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(arr_rgb)
                inp = detr_proc(images=pil, return_tensors='pt').to(DEVICE)

                torch.cuda.synchronize()
                t_m0 = time.perf_counter()
                out = detr(**inp)
                torch.cuda.synchronize()
                t_m1 = time.perf_counter()

                _ = detr_proc.post_process_object_detection(out, target_sizes=torch.tensor([pil.size[::-1]]).to(DEVICE), threshold=0.05)[0]
                torch.cuda.synchronize()
                t1_e2e = time.perf_counter()

                mod_times.append(float((t_m1 - t_m0) * 1000.0))
                e2e_times.append(float((t1_e2e - t0_e2e) * 1000.0))

    elif model_name == 'efficientdet':
        from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

        sz = (640, 640) if mode == 'uniform640' else (512, 512)
        cfg = get_efficientdet_config('tf_efficientdet_d2')
        cfg.num_classes = 4
        cfg.image_size = sz
        net = EfficientDet(cfg, pretrained_backbone=False)
        net.load_state_dict(torch.load('checkpoints/efficientdet_best.pt', map_location=DEVICE))
        effdet = DetBenchPredict(net).to(DEVICE).eval()

        # Warmup (25 images)
        for _, b in raw_imgs[:25]:
            arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            res = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), sz)
            norm = (res / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            t = torch.tensor(norm, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
            _ = effdet(t, img_info={'img_scale': torch.tensor([1.0], device=DEVICE), 'img_size': torch.tensor([list(sz)], device=DEVICE)})
        torch.cuda.synchronize()

        mod_times = []
        e2e_times = []
        with torch.no_grad():
            for _, b in raw_imgs:
                torch.cuda.synchronize()
                t0_e2e = time.perf_counter()
                arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                res = cv2.resize(arr_rgb, sz)
                norm = (res / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                tens = torch.tensor(norm, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)

                torch.cuda.synchronize()
                t_m0 = time.perf_counter()
                out = effdet(tens, img_info={'img_scale': torch.tensor([1.0], device=DEVICE), 'img_size': torch.tensor([list(sz)], device=DEVICE)})[0]
                torch.cuda.synchronize()
                t_m1 = time.perf_counter()

                _ = out[:, 0:4].cpu()
                torch.cuda.synchronize()
                t1_e2e = time.perf_counter()

                mod_times.append(float((t_m1 - t_m0) * 1000.0))
                e2e_times.append(float((t1_e2e - t0_e2e) * 1000.0))

    # Compute statistics
    mod_arr = np.array(mod_times)
    e2e_arr = np.array(e2e_times)

    summary = {
        'model': model_name,
        'mode': mode,
        'repeat': args.repeat,
        'mod_median': float(np.median(mod_arr)),
        'mod_iqr': float(np.percentile(mod_arr, 75) - np.percentile(mod_arr, 25)),
        'e2e_median': float(np.median(e2e_arr)),
        'e2e_iqr': float(np.percentile(e2e_arr, 75) - np.percentile(e2e_arr, 25)),
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[{model_name.upper()} {mode} R{args.repeat}] Mod Median: {summary['mod_median']:.2f} ms | E2E Median: {summary['e2e_median']:.2f} ms")

if __name__ == '__main__':
    main()
