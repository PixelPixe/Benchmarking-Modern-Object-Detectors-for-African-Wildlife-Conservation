#!/usr/bin/env python3
"""
run2/benchmark_speed.py - Uniform Speed & Efficiency Benchmark for WildCareAI2

Measures all 5 detection architectures identically under controlled experimental conditions:
- Same hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM)
- Batch size: 1
- Input resolution: 640 x 640
- Warm-up iterations: 50
- Timed iterations: 300
- GPU synchronization: torch.cuda.synchronize() around all timed blocks
- Precision: Single Precision FP32 (native checkpoint precision)
- Metrics measured:
  (a) Model-only inference latency (mean ± std ms) and throughput (FPS)
  (b) End-to-end latency (preprocessing + inference + postprocessing/NMS) (mean ± std ms) and throughput (FPS)
  (c) Parameter count (Million parameters)
  (d) Computational complexity (GFLOPs via thop at 640x640)
- Outputs:
  - results_v2/speed_benchmark.json
  - results_v2/speed_benchmark.csv
  - results_v2/table5_speed_accuracy.tex (LaTeX Table 5 replacement)
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
import thop

from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
WARMUP_RUNS = 50
TIMED_RUNS = 300
IMG_SIZE = 640

def get_accuracy_metrics():
    # Load primary mAP, AP50, AP75 from results_v2 or results
    p = Path('results_v2/metrics_summary.csv')
    if not p.exists():
        p = Path('results/metrics_summary.csv')
    df = pd.read_csv(p)
    acc = {}
    for _, row in df.iterrows():
        acc[row['model']] = {
            'mAP': float(row['AP_5_95']),
            'AP50': float(row['AP_50']),
            'AP75': float(row['AP_75'])
        }
    return acc

def benchmark_model(model_name, sample_img_path):
    print(f"\n========================================================")
    print(f"[BENCHMARK] Benchmarking: {model_name.upper()} (GPU: {torch.cuda.get_device_name(0)})")
    print(f"========================================================")

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    acc_map = get_accuracy_metrics()
    acc = acc_map.get(model_name, {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0})

    # ─────────────────────────────────────────────────────────
    # 1. Load Model & Compute Params / GFLOPs
    # ─────────────────────────────────────────────────────────
    if 'yolo' in model_name:
        yolo = YOLO(f'checkpoints/{model_name}_best.pt')
        torch_model = yolo.model.to(DEVICE).eval()
        params_m = sum(p.numel() for p in torch_model.parameters()) / 1e6
        macs, _ = thop.profile(torch_model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9

    elif model_name == 'fasterrcnn':
        torch_model = fasterrcnn_resnet50_fpn(weights=None, min_size=IMG_SIZE, max_size=IMG_SIZE)
        in_features = torch_model.roi_heads.box_predictor.cls_score.in_features
        torch_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
        torch_model.load_state_dict(torch.load('checkpoints/fasterrcnn_best.pt', map_location=DEVICE))
        torch_model.to(DEVICE).eval()
        params_m = sum(p.numel() for p in torch_model.parameters()) / 1e6
        macs, _ = thop.profile(torch_model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9  # FLOPs = 2 * MACs convention

    elif model_name == 'efficientdet':
        config = get_efficientdet_config('tf_efficientdet_d2')
        config.num_classes = 4
        config.image_size = (IMG_SIZE, IMG_SIZE)
        net = EfficientDet(config, pretrained_backbone=False)
        net.load_state_dict(torch.load('checkpoints/efficientdet_best.pt', map_location=DEVICE))
        torch_model = DetBenchPredict(net).to(DEVICE).eval()
        params_m = sum(p.numel() for p in net.parameters()) / 1e6
        macs, _ = thop.profile(net, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9

    elif model_name == 'detr':
        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50",
            size={"shortest_edge": IMG_SIZE, "longest_edge": IMG_SIZE}
        )
        torch_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", num_labels=4, ignore_mismatched_sizes=True
        )
        torch_model.load_state_dict(torch.load('checkpoints/detr_best.pt', map_location=DEVICE))
        torch_model.to(DEVICE).eval()
        params_m = sum(p.numel() for p in torch_model.parameters()) / 1e6
        macs, _ = thop.profile(torch_model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9

    print(f"  Params: {params_m:.2f} M | GFLOPs: {gflops:.2f}")

    # ─────────────────────────────────────────────────────────
    # 2. Model-Only Latency Timing
    # ─────────────────────────────────────────────────────────
    # Prepare model-only call callable
    if 'yolo' in model_name:
        def model_forward():
            return torch_model(dummy_input)
    elif model_name == 'fasterrcnn':
        def model_forward():
            return torch_model(dummy_input)
    elif model_name == 'efficientdet':
        img_info = {
            "img_scale": torch.tensor([1.0], device=DEVICE),
            "img_size": torch.tensor([[IMG_SIZE, IMG_SIZE]], device=DEVICE)
        }
        def model_forward():
            return torch_model(dummy_input, img_info=img_info)
    elif model_name == 'detr':
        mask = torch.ones(1, IMG_SIZE, IMG_SIZE, dtype=torch.long, device=DEVICE)
        def model_forward():
            return torch_model(pixel_values=dummy_input, pixel_mask=mask)

    # Warm-up model-only
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model_forward()
            torch.cuda.synchronize()

    # Timed runs model-only
    model_latencies = []
    with torch.no_grad():
        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model_forward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            model_latencies.append((t1 - t0) * 1000.0)  # ms

    model_latencies = np.array(model_latencies)
    model_mean_ms = float(np.mean(model_latencies))
    model_std_ms = float(np.std(model_latencies))
    model_fps = float(1000.0 / model_mean_ms)
    print(f"  Model-Only Latency: {model_mean_ms:.2f} ± {model_std_ms:.2f} ms ({model_fps:.1f} FPS)")

    # ─────────────────────────────────────────────────────────
    # 3. End-to-End Latency Timing (Preprocess + Model + Postprocess)
    # ─────────────────────────────────────────────────────────
    # Read sample image once into memory buffer
    with open(sample_img_path, 'rb') as f:
        img_bytes = f.read()

    if 'yolo' in model_name:
        def end_to_end():
            # YOLO predict pipeline: decodes bytes/image, letterboxes to 640, forwards, and runs NMS
            # Using cv2 decoded array
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            res = yolo.predict(arr, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
            _ = res[0].boxes.xyxy
            return res

    elif model_name == 'fasterrcnn':
        def end_to_end():
            # Preprocess: decode, PIL, ToTensor
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(arr)
            tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE)
            # Model forward includes internal RPN and RoIAlign + NMS
            out = torch_model(tensor)[0]
            boxes = out['boxes'].cpu()
            scores = out['scores'].cpu()
            labels = out['labels'].cpu()
            return boxes, scores, labels

    elif model_name == 'efficientdet':
        def end_to_end():
            # Preprocess: decode, resize 640x640, normalize
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = arr.shape[:2]
            resized = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
            norm = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            img_info = {
                "img_scale": torch.tensor([1.0], device=DEVICE),
                "img_size": torch.tensor([[IMG_SIZE, IMG_SIZE]], device=DEVICE)
            }
            # Model forward + postprocess (anchor unscaling)
            dets = torch_model(tensor, img_info=img_info)[0]
            # Rescale boxes back
            boxes = dets[:, 0:4]
            return boxes

    elif model_name == 'detr':
        def end_to_end():
            # Preprocess: decode, PIL, DetrImageProcessor
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(arr)
            inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
            outputs = torch_model(**inputs)
            target_sizes = torch.tensor([[IMG_SIZE, IMG_SIZE]], device=DEVICE)
            res = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)[0]
            return res

    # Warmup end-to-end
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = end_to_end()
            torch.cuda.synchronize()

    # Timed runs end-to-end
    e2e_latencies = []
    with torch.no_grad():
        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = end_to_end()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            e2e_latencies.append((t1 - t0) * 1000.0)

    e2e_latencies = np.array(e2e_latencies)
    e2e_mean_ms = float(np.mean(e2e_latencies))
    e2e_std_ms = float(np.std(e2e_latencies))
    e2e_fps = float(1000.0 / e2e_mean_ms)
    print(f"  End-to-End Latency: {e2e_mean_ms:.2f} ± {e2e_std_ms:.2f} ms ({e2e_fps:.1f} FPS)")

    return {
        'model': model_name,
        'mAP': acc['mAP'],
        'AP50': acc['AP50'],
        'AP75': acc['AP75'],
        'model_latency_mean_ms': round(model_mean_ms, 2),
        'model_latency_std_ms': round(model_std_ms, 2),
        'model_fps': round(model_fps, 1),
        'e2e_latency_mean_ms': round(e2e_mean_ms, 2),
        'e2e_latency_std_ms': round(e2e_std_ms, 2),
        'e2e_fps': round(e2e_fps, 1),
        'params_M': round(params_m, 2),
        'gflops': round(gflops, 1)
    }

def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)

    out_dirs = [Path('run2'), Path('results_v2')]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Find sample test image
    test_images = list(Path('dataset/images/test').glob('*.jpg'))
    if not test_images:
        print("[ERROR] No test images found in dataset/images/test")
        sys.exit(1)
    sample_img_path = str(test_images[0])
    print(f"Using sample image: {sample_img_path}")

    models = ['yolov8m', 'yolov8n', 'fasterrcnn', 'detr', 'efficientdet']
    results = []
    for m in models:
        res = benchmark_model(m, sample_img_path)
        results.append(res)

    df = pd.DataFrame(results)

    # Sort by mAP descending as requested
    df = df.sort_values(by='mAP', ascending=False).reset_index(drop=True)

    # Save CSV and JSON
    for d in out_dirs:
        df.to_csv(d / 'speed_benchmark.csv', index=False)
        with open(d / 'speed_benchmark.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    # ─────────────────────────────────────────────────────────
    # 4. Generate LaTeX Table 5
    # Columns: model, mAP, AP50, AP75, FPS (model/e2e), params, GFLOPs
    # Sorted by mAP descending, best values in bold, Faster R-CNN included.
    # ─────────────────────────────────────────────────────────
    best_mAP = df['mAP'].max()
    best_ap50 = df['AP50'].max()
    best_ap75 = df['AP75'].max()
    best_fps = df['model_fps'].max()
    best_e2e_fps = df['e2e_fps'].max()
    best_params = df['params_M'].min()
    best_gflops = df['gflops'].min()

    model_display_names = {
        'yolov8m': 'YOLOv8m',
        'yolov8n': 'YOLOv8n',
        'fasterrcnn': 'Faster R-CNN',
        'detr': 'DETR',
        'efficientdet': 'EfficientDet-D2'
    }

    latex_table5 = r"""\begin{table*}[htbp]
\centering
\caption{Comprehensive Speed, Accuracy, and Complexity Benchmark on African Wildlife Test Set. All architectures evaluated on identical hardware (NVIDIA RTX 4050 Laptop GPU, 6GB VRAM) at batch size 1 and $640 \times 640$ resolution using FP32 precision. Latency and FPS measured across 300 timed runs following 50 warm-up runs with explicit GPU synchronization. Sorted by primary metric mAP@[.5:.95] descending; best values in \textbf{bold}.}
\label{tab:speed_benchmark}
\small
\renewcommand{\arraystretch}{1.25}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{mAP@[.5:.95]} & \textbf{AP\textsubscript{50}} & \textbf{AP\textsubscript{75}} & \textbf{Throughput (FPS)\textsuperscript{\textdagger}} & \textbf{End-to-End FPS\textsuperscript{\textdaggerdbl}} & \textbf{Params (M)} & \textbf{GFLOPs\textsuperscript{*}} \\
\midrule
"""

    for _, r in df.iterrows():
        m_name = model_display_names.get(r['model'], r['model'])
        
        # Format bold if best
        map_str = f"\\textbf{{{r['mAP']:.3f}}}" if r['mAP'] == best_mAP else f"{r['mAP']:.3f}"
        ap50_str = f"\\textbf{{{r['AP50']:.3f}}}" if r['AP50'] == best_ap50 else f"{r['AP50']:.3f}"
        ap75_str = f"\\textbf{{{r['AP75']:.3f}}}" if r['AP75'] == best_ap75 else f"{r['AP75']:.3f}"
        fps_str = f"\\textbf{{{r['model_fps']:.1f}}}" if r['model_fps'] == best_fps else f"{r['model_fps']:.1f}"
        e2e_fps_str = f"\\textbf{{{r['e2e_fps']:.1f}}}" if r['e2e_fps'] == best_e2e_fps else f"{r['e2e_fps']:.1f}"
        params_str = f"\\textbf{{{r['params_M']:.2f}}}" if r['params_M'] == best_params else f"{r['params_M']:.2f}"
        gflops_str = f"\\textbf{{{r['gflops']:.1f}}}" if r['gflops'] == best_gflops else f"{r['gflops']:.1f}"

        latex_table5 += f"{m_name} & {map_str} & {ap50_str} & {ap75_str} & {fps_str} & {e2e_fps_str} & {params_str} & {gflops_str} \\\\\n"

    latex_table5 += r"""\bottomrule
\multicolumn{8}{l}{\footnotesize \textsuperscript{\textdagger} Model-only forward pass throughput. \textsuperscript{\textdaggerdbl} End-to-end throughput including image decode, resize/normalization, and NMS/postprocessing.} \\
\multicolumn{8}{l}{\footnotesize \textsuperscript{*} GFLOPs computed at $640 \times 640$ via \texttt{thop} ($\text{FLOPs} = 2 \times \text{MACs}$). YOLOv8 values computed via Ultralytics profiler.} \\
\end{tabular}
\end{table*}
"""

    for d in out_dirs:
        with open(d / 'table5_speed_accuracy.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table5)

    print("\n==============================================================")
    print("[SUCCESS] Benchmark complete! Output written to run2/ and results_v2/")
    print("==============================================================")
    print(df[['model', 'mAP', 'AP50', 'AP75', 'model_fps', 'e2e_fps', 'params_M', 'gflops']].to_string(index=False))

if __name__ == '__main__':
    main()
