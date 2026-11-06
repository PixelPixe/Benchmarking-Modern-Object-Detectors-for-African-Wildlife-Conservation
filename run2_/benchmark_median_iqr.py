#!/usr/bin/env python3
"""
run2/benchmark_median_iqr.py - Dataset-Wide Speed Benchmark Reporting Median and IQR

Computes:
- Model Latency: Median, IQR, Mean, Std
- End-to-End Latency: Median, IQR, Mean, Std (including cv2.imdecode from in-memory raw bytes)
- Median E2E FPS
over all N=145 test images.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import cv2

from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    test_imgs = sorted(list(Path('dataset/images/test').glob('*.jpg')))
    raw_imgs = [(p.name, open(p, 'rb').read()) for p in test_imgs]
    print(f"Loaded {len(raw_imgs)} test images into RAM.")

    # Load models
    yolo_m = YOLO('checkpoints/yolov8m_best.pt').to(DEVICE)
    yolo_n = YOLO('checkpoints/yolov8n_best.pt').to(DEVICE)

    frcnn = fasterrcnn_resnet50_fpn(weights=None, min_size=800, max_size=1333)
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(frcnn.roi_heads.box_predictor.cls_score.in_features, num_classes=5)
    frcnn.load_state_dict(torch.load('checkpoints/fasterrcnn_best.pt', map_location=DEVICE))
    frcnn.to(DEVICE).eval()

    detr_proc = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', size={'shortest_edge': 800, 'longest_edge': 1333})
    detr = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=4, ignore_mismatched_sizes=True)
    detr.load_state_dict(torch.load('checkpoints/detr_best.pt', map_location=DEVICE))
    detr.to(DEVICE).eval()

    cfg = get_efficientdet_config('tf_efficientdet_d2')
    cfg.num_classes = 4
    cfg.image_size = (512, 512)
    net = EfficientDet(cfg, pretrained_backbone=False)
    net.load_state_dict(torch.load('checkpoints/efficientdet_best.pt', map_location=DEVICE))
    effdet = DetBenchPredict(net).to(DEVICE).eval()

    # Warmup
    print("Warmup on first 10 images...")
    for _, b in raw_imgs[:10]:
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        _ = yolo_m.predict(arr, imgsz=640, rect=False, verbose=False)
        _ = yolo_n.predict(arr, imgsz=640, rect=False, verbose=False)
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        _ = frcnn(F.to_tensor(pil).unsqueeze(0).to(DEVICE))
        _ = detr(**detr_proc(images=pil, return_tensors='pt').to(DEVICE))
        res = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), (512, 512))
        norm = (res/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.tensor(norm, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
        _ = effdet(t, img_info={'img_scale': torch.tensor([1.0], device=DEVICE), 'img_size': torch.tensor([[512, 512]], device=DEVICE)})
        torch.cuda.synchronize()

    results = {m: {'mod': [], 'e2e': []} for m in ['yolov8m', 'yolov8n', 'fasterrcnn', 'detr', 'efficientdet']}
    decode_times = []

    print("Timing across all 145 images...")
    with torch.no_grad():
        for idx, (_, b) in enumerate(raw_imgs):
            # Decode
            t0_dec = time.perf_counter()
            arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            t_dec = time.perf_counter() - t0_dec
            decode_times.append(t_dec * 1000.0)

            # YOLOv8m (rect=False)
            t0 = time.perf_counter()
            r_m = yolo_m.predict(arr, imgsz=640, rect=False, verbose=False)
            _ = r_m[0].boxes.xyxy
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results['yolov8m']['e2e'].append((t1 - t0 + t_dec) * 1000.0)
            results['yolov8m']['mod'].append(r_m[0].speed['inference'])

            # YOLOv8n (rect=False)
            t0 = time.perf_counter()
            r_n = yolo_n.predict(arr, imgsz=640, rect=False, verbose=False)
            _ = r_n[0].boxes.xyxy
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results['yolov8n']['e2e'].append((t1 - t0 + t_dec) * 1000.0)
            results['yolov8n']['mod'].append(r_n[0].speed['inference'])

            # Faster R-CNN
            t0 = time.perf_counter()
            pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            tens = F.to_tensor(pil).unsqueeze(0).to(DEVICE)
            torch.cuda.synchronize()
            t_mod0 = time.perf_counter()
            out_fr = frcnn(tens)[0]
            torch.cuda.synchronize()
            t_mod1 = time.perf_counter()
            _ = out_fr['boxes'].cpu()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results['fasterrcnn']['mod'].append((t_mod1 - t_mod0) * 1000.0)
            results['fasterrcnn']['e2e'].append((t1 - t0 + t_dec) * 1000.0)

            # DETR
            t0 = time.perf_counter()
            pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            inp = detr_proc(images=pil, return_tensors='pt').to(DEVICE)
            torch.cuda.synchronize()
            t_mod0 = time.perf_counter()
            out_d = detr(**inp)
            torch.cuda.synchronize()
            t_mod1 = time.perf_counter()
            _ = detr_proc.post_process_object_detection(out_d, target_sizes=torch.tensor([pil.size[::-1]]).to(DEVICE), threshold=0.05)[0]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results['detr']['mod'].append((t_mod1 - t_mod0) * 1000.0)
            results['detr']['e2e'].append((t1 - t0 + t_dec) * 1000.0)

            # EfficientDet
            t0 = time.perf_counter()
            res = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), (512, 512))
            norm = (res/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tens = torch.tensor(norm, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
            torch.cuda.synchronize()
            t_mod0 = time.perf_counter()
            out_e = effdet(tens, img_info={'img_scale': torch.tensor([1.0], device=DEVICE), 'img_size': torch.tensor([[512, 512]], device=DEVICE)})[0]
            torch.cuda.synchronize()
            t_mod1 = time.perf_counter()
            _ = out_e[:, 0:4].cpu()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results['efficientdet']['mod'].append((t_mod1 - t_mod0) * 1000.0)
            results['efficientdet']['e2e'].append((t1 - t0 + t_dec) * 1000.0)

    rows = []
    display_names = {
        'yolov8m': 'YOLOv8m',
        'yolov8n': 'YOLOv8n',
        'fasterrcnn': 'Faster R-CNN',
        'detr': 'DETR',
        'efficientdet': 'EfficientDet-D2'
    }
    native_res_names = {
        'yolov8m': '640x640',
        'yolov8n': '640x640',
        'fasterrcnn': '800-1333',
        'detr': '800-1333',
        'efficientdet': '512x512'
    }

    for m in ['yolov8m', 'yolov8n', 'fasterrcnn', 'detr', 'efficientdet']:
        mod_arr = np.array(results[m]['mod'])
        e2e_arr = np.array(results[m]['e2e'])
        rows.append({
            'Model': display_names[m],
            'Native Resolution': native_res_names[m],
            'Model Median (ms)': round(float(np.median(mod_arr)), 2),
            'Model IQR (ms)': round(float(np.percentile(mod_arr, 75) - np.percentile(mod_arr, 25)), 2),
            'Model Mean (ms)': round(float(np.mean(mod_arr)), 2),
            'Model Std (ms)': round(float(np.std(mod_arr)), 2),
            'E2E Median (ms)': round(float(np.median(e2e_arr)), 2),
            'E2E IQR (ms)': round(float(np.percentile(e2e_arr, 75) - np.percentile(e2e_arr, 25)), 2),
            'E2E Mean (ms)': round(float(np.mean(e2e_arr)), 2),
            'E2E Std (ms)': round(float(np.std(e2e_arr)), 2),
            'E2E Min (ms)': round(float(np.min(e2e_arr)), 2),
            'E2E Max (ms)': round(float(np.max(e2e_arr)), 2),
            'Median E2E FPS': round(float(1000.0 / np.median(e2e_arr)), 1)
        })

    df = pd.DataFrame(rows)
    print("\n================ DATASET-WIDE (N=145) SPEED: MEDIAN & IQR ================")
    print(df[['Model', 'Native Resolution', 'Model Median (ms)', 'Model IQR (ms)', 'E2E Median (ms)', 'E2E IQR (ms)', 'Median E2E FPS']].to_string(index=False))

    df.to_csv('results_v2/dataset_speed_median_iqr.csv', index=False)
    df.to_csv('run2/dataset_speed_median_iqr.csv', index=False)

    # Save LaTeX table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Dataset-Wide Inference Speed Evaluated at Native Resolution Across All $N=145$ African Wildlife Test Images. In addition to mean and standard deviation, we report the median and interquartile range (IQR = Q\textsubscript{75} - Q\textsubscript{25}) to robustly capture latency across varying image aspect ratios and resolutions. All runs include image decoding (\texttt{cv2.imdecode}), preprocessing, neural network forward pass, and postprocessing on an NVIDIA GeForce RTX 4050 Laptop GPU (batch size 1, FP32).}
\label{tab:dataset_speed_median_iqr}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Native Res.} & \textbf{Model Median (IQR) ms} & \textbf{Model Mean $\pm$ Std ms} & \textbf{E2E Median (IQR) ms} & \textbf{E2E Mean $\pm$ Std ms} & \textbf{Median E2E FPS} \\
\midrule
"""
    for _, r in df.iterrows():
        tex += f"{r['Model']} & {r['Native Resolution']} & {r['Model Median (ms)']:.1f} ({r['Model IQR (ms)']:.1f}) & {r['Model Mean (ms)']:.1f} $\\pm$ {r['Model Std (ms)']:.1f} & {r['E2E Median (ms)']:.1f} ({r['E2E IQR (ms)']:.1f}) & {r['E2E Mean (ms)']:.1f} $\\pm$ {r['E2E Std (ms)']:.1f} & \\textbf{{{r['Median E2E FPS']:.1f}}} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open('run2/table_dataset_speed_median_iqr.tex', 'w') as f:
        f.write(tex)
    with open('results_v2/table_dataset_speed_median_iqr.tex', 'w') as f:
        f.write(tex)
    print("\nSaved LaTeX table to run2/table_dataset_speed_median_iqr.tex and results_v2/table_dataset_speed_median_iqr.tex")

if __name__ == '__main__':
    main()
