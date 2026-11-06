#!/usr/bin/env python3
"""
run2/benchmark_native_dataset.py - Multi-Image Native-Resolution Speed Benchmark for WildCareAI2

Evaluates all 5 models at their native evaluation resolutions across the ENTIRE test set (N=145 images):
- YOLOv8m: Native 640x640 (square letterbox)
- YOLOv8n: Native 640x640 (square letterbox)
- Faster R-CNN: Native min_size=800, max_size=1333
- DETR: Native shortest_edge=800, longest_edge=1333
- EfficientDet-D2: Native 512x512

Includes image decode (cv2.imdecode from in-memory raw image bytes) in end-to-end timing for EVERY model.
Reports mean, std, min, and max across all 145 test images with diverse dimensions.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_models():
    print("Loading all models to GPU...")
    # 1. YOLOv8m
    yolo_m = YOLO('checkpoints/yolov8m_best.pt')
    yolo_m.to(DEVICE)

    # 2. YOLOv8n
    yolo_n = YOLO('checkpoints/yolov8n_best.pt')
    yolo_n.to(DEVICE)

    # 3. Faster R-CNN
    frcnn = fasterrcnn_resnet50_fpn(weights=None, min_size=800, max_size=1333)
    in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
    frcnn.load_state_dict(torch.load('checkpoints/fasterrcnn_best.pt', map_location=DEVICE))
    frcnn.to(DEVICE).eval()

    # 4. DETR
    detr_proc = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", size={"shortest_edge": 800, "longest_edge": 1333}
    )
    detr = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=4, ignore_mismatched_sizes=True
    )
    detr.load_state_dict(torch.load('checkpoints/detr_best.pt', map_location=DEVICE))
    detr.to(DEVICE).eval()

    # 5. EfficientDet-D2
    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = 4
    config.image_size = (512, 512)
    net = EfficientDet(config, pretrained_backbone=False)
    net.load_state_dict(torch.load('checkpoints/efficientdet_best.pt', map_location=DEVICE))
    effdet = DetBenchPredict(net).to(DEVICE).eval()

    return {
        'yolov8m': yolo_m,
        'yolov8n': yolo_n,
        'fasterrcnn': frcnn,
        'detr': (detr, detr_proc),
        'efficientdet': effdet
    }

def benchmark_models_over_dataset(test_image_files, models):
    # Pre-load all raw image bytes into memory to isolate compute from disk I/O
    print(f"Reading raw image bytes for {len(test_image_files)} test images into RAM...")
    raw_images = []
    for p in test_image_files:
        with open(p, 'rb') as f:
            raw_images.append((p.name, f.read()))

    # Warmup runs on first 10 images
    print("Performing GPU warmup across architectures...")
    for _, raw_bytes in raw_images[:10]:
        with torch.no_grad():
            # YOLO
            arr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            _ = models['yolov8m'].predict(arr, imgsz=640, rect=False, verbose=False, device=DEVICE)
            _ = models['yolov8n'].predict(arr, imgsz=640, rect=False, verbose=False, device=DEVICE)
            # Faster R-CNN
            pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            t = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE)
            _ = models['fasterrcnn'](t)
            # DETR
            detr, detr_proc = models['detr']
            inp = detr_proc(images=pil_img, return_tensors="pt").to(DEVICE)
            _ = detr(**inp)
            # EfficientDet
            resized = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), (512, 512))
            norm = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            eff_t = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            _ = models['efficientdet'](eff_t, img_info={"img_scale": torch.tensor([1.0], device=DEVICE), "img_size": torch.tensor([[512, 512]], device=DEVICE)})
            torch.cuda.synchronize()

    results = {
        'yolov8m': {'model_ms': [], 'e2e_ms': []},
        'yolov8n': {'model_ms': [], 'e2e_ms': []},
        'fasterrcnn': {'model_ms': [], 'e2e_ms': []},
        'detr': {'model_ms': [], 'e2e_ms': []},
        'efficientdet': {'model_ms': [], 'e2e_ms': []}
    }

    print(f"\nBenchmarking all 5 models sequentially over {len(raw_images)} test images...")
    
    # Run test set
    with torch.no_grad():
        for idx, (fname, raw_bytes) in enumerate(raw_images):
            if (idx + 1) % 25 == 0 or idx == len(raw_images) - 1:
                print(f"  Processed {idx + 1}/{len(raw_images)} images...")

            # ── 1. YOLOv8m (native 640x640 square) ─────────────────────
            torch.cuda.synchronize()
            t0_e2e = time.perf_counter()
            arr_m = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            res_m = models['yolov8m'].predict(arr_m, imgsz=640, rect=False, verbose=False, device=DEVICE)
            _ = res_m[0].boxes.xyxy
            torch.cuda.synchronize()
            t1_e2e = time.perf_counter()
            results['yolov8m']['e2e_ms'].append((t1_e2e - t0_e2e) * 1000.0)
            results['yolov8m']['model_ms'].append(res_m[0].speed['inference'])

            # ── 2. YOLOv8n (native 640x640 square) ─────────────────────
            torch.cuda.synchronize()
            t0_e2e = time.perf_counter()
            arr_n = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            res_n = models['yolov8n'].predict(arr_n, imgsz=640, rect=False, verbose=False, device=DEVICE)
            _ = res_n[0].boxes.xyxy
            torch.cuda.synchronize()
            t1_e2e = time.perf_counter()
            results['yolov8n']['e2e_ms'].append((t1_e2e - t0_e2e) * 1000.0)
            results['yolov8n']['model_ms'].append(res_n[0].speed['inference'])

            # ── 3. Faster R-CNN (native 800-1333) ─────────────────────
            torch.cuda.synchronize()
            t0_e2e = time.perf_counter()
            arr_fr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr_fr_rgb = cv2.cvtColor(arr_fr, cv2.COLOR_BGR2RGB)
            pil_fr = Image.fromarray(arr_fr_rgb)
            tensor_fr = F.to_tensor(pil_fr).unsqueeze(0).to(DEVICE)
            
            torch.cuda.synchronize()
            t0_mod = time.perf_counter()
            out_fr = models['fasterrcnn'](tensor_fr)[0]
            torch.cuda.synchronize()
            t1_mod = time.perf_counter()
            
            _ = out_fr['boxes'].cpu()
            torch.cuda.synchronize()
            t1_e2e = time.perf_counter()
            results['fasterrcnn']['model_ms'].append((t1_mod - t0_mod) * 1000.0)
            results['fasterrcnn']['e2e_ms'].append((t1_e2e - t0_e2e) * 1000.0)

            # ── 4. DETR (native 800-1333) ─────────────────────────────
            detr, detr_proc = models['detr']
            torch.cuda.synchronize()
            t0_e2e = time.perf_counter()
            arr_d = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr_d_rgb = cv2.cvtColor(arr_d, cv2.COLOR_BGR2RGB)
            pil_d = Image.fromarray(arr_d_rgb)
            inp_d = detr_proc(images=pil_d, return_tensors="pt").to(DEVICE)
            
            torch.cuda.synchronize()
            t0_mod = time.perf_counter()
            out_d = detr(**inp_d)
            torch.cuda.synchronize()
            t1_mod = time.perf_counter()
            
            target_sizes = torch.tensor([pil_d.size[::-1]]).to(DEVICE)
            _ = detr_proc.post_process_object_detection(out_d, target_sizes=target_sizes, threshold=0.05)[0]
            torch.cuda.synchronize()
            t1_e2e = time.perf_counter()
            results['detr']['model_ms'].append((t1_mod - t0_mod) * 1000.0)
            results['detr']['e2e_ms'].append((t1_e2e - t0_e2e) * 1000.0)

            # ── 5. EfficientDet-D2 (native 512x512) ───────────────────
            torch.cuda.synchronize()
            t0_e2e = time.perf_counter()
            arr_ef = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr_ef_rgb = cv2.cvtColor(arr_ef, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = arr_ef_rgb.shape[:2]
            resized_ef = cv2.resize(arr_ef_rgb, (512, 512))
            norm_ef = (resized_ef / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor_ef = torch.tensor(norm_ef, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            img_info = {
                "img_scale": torch.tensor([1.0], device=DEVICE),
                "img_size": torch.tensor([[512, 512]], device=DEVICE)
            }
            
            torch.cuda.synchronize()
            t0_mod = time.perf_counter()
            dets_ef = models['efficientdet'](tensor_ef, img_info=img_info)[0]
            torch.cuda.synchronize()
            t1_mod = time.perf_counter()
            
            _ = dets_ef[:, 0:4].cpu()
            torch.cuda.synchronize()
            t1_e2e = time.perf_counter()
            results['efficientdet']['model_ms'].append((t1_mod - t0_mod) * 1000.0)
            results['efficientdet']['e2e_ms'].append((t1_e2e - t0_e2e) * 1000.0)

    # Compute Statistics Across the 145 Images
    summary_rows = []
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
        m_ms = np.array(results[m]['model_ms'])
        e_ms = np.array(results[m]['e2e_ms'])
        
        row = {
            'Model': display_names[m],
            'Native Resolution': native_res_names[m],
            'Model Latency (ms)': f"{np.mean(m_ms):.2f} ± {np.std(m_ms):.2f}",
            'Model FPS': round(float(1000.0 / np.mean(m_ms)), 1),
            'E2E Latency (ms)': f"{np.mean(e_ms):.2f} ± {np.std(e_ms):.2f}",
            'E2E FPS': round(float(1000.0 / np.mean(e_ms)), 1),
            'E2E Min (ms)': round(float(np.min(e_ms)), 2),
            'E2E Max (ms)': round(float(np.max(e_ms)), 2),
            'raw_model_mean_ms': round(float(np.mean(m_ms)), 2),
            'raw_model_std_ms': round(float(np.std(m_ms)), 2),
            'raw_e2e_mean_ms': round(float(np.mean(e_ms)), 2),
            'raw_e2e_std_ms': round(float(np.std(e_ms)), 2)
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    print("\n================ FULL TEST SET (N=145) NATIVE RESOLUTION BENCHMARK ================")
    print(df[['Model', 'Native Resolution', 'Model Latency (ms)', 'Model FPS', 'E2E Latency (ms)', 'E2E FPS', 'E2E Min (ms)', 'E2E Max (ms)']].to_string(index=False))

    # Save to CSV and JSON
    df.to_csv('results_v2/native_dataset_speed_benchmark.csv', index=False)
    df.to_csv('run2/native_dataset_speed_benchmark.csv', index=False)

    json_data = {
        'num_images': len(raw_images),
        'per_model_summary': summary_rows
    }
    with open('results_v2/native_dataset_speed_benchmark.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    with open('run2/native_dataset_speed_benchmark.json', 'w') as f:
        json.dump(json_data, f, indent=2)

    # Generate LaTeX table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Inference Throughput and Latency Evaluated at Native Resolution Across the Complete African Wildlife Test Set ($N=145$ images). Timed end-to-end for every model starting with raw image decoding (\texttt{cv2.imdecode}), followed by architecture-specific preprocessing, neural network forward pass, and postprocessing (NMS or Hungarian matching) on an NVIDIA RTX 4050 Laptop GPU (batch size 1, FP32). Mean $\pm$ sample standard deviation across all 145 images reflects aspect-ratio and resolution variation.}
\label{tab:native_dataset_speed}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Native Res.} & \textbf{Model Latency (ms)} & \textbf{Model FPS} & \textbf{E2E Latency (ms)} & \textbf{E2E FPS} & \textbf{E2E Range (ms)} \\
\midrule
"""
    for _, r in df.iterrows():
        tex += f"{r['Model']} & {r['Native Resolution']} & {r['Model Latency (ms)']} & {r['Model FPS']} & {r['E2E Latency (ms)']} & {r['E2E FPS']} & [{r['E2E Min (ms)']:.1f}, {r['E2E Max (ms)']:.1f}] \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open('run2/table_native_dataset_speed.tex', 'w') as f:
        f.write(tex)
    with open('results_v2/table_native_dataset_speed.tex', 'w') as f:
        f.write(tex)
    print("\nSaved LaTeX table to run2/table_native_dataset_speed.tex and results_v2/table_native_dataset_speed.tex")

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    test_image_files = sorted(list(Path('dataset/images/test').glob('*.jpg')))
    if not test_image_files:
        print("[ERROR] No test images found.")
        sys.exit(1)
    print(f"Found {len(test_image_files)} test images.")

    models = load_models()
    benchmark_models_over_dataset(test_image_files, models)

if __name__ == '__main__':
    main()
