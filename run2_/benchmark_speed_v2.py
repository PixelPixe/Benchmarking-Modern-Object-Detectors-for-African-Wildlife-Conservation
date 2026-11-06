#!/usr/bin/env python3
"""
run2/benchmark_speed_v2.py - Comprehensive Speed, Efficiency & Resolution Benchmark for WildCareAI2

Addresses peer review feedback:
1. Native vs. Uniform 640x640 resolution benchmarks:
   - Native: YOLOv8 (640x640), Faster R-CNN (800-1333), DETR (800-1333), EfficientDet-D2 (512x512)
   - Uniform: 640x640 for all models
2. End-to-end breakdown and investigation of YOLOv8n vs YOLOv8m overhead.
3. Cross-check GFLOPs with fvcore for DETR and Faster R-CNN (noting SDPA and RoIAlign).
4. GPU telemetry logging (clocks, temp, power mode).
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
import thop
from fvcore.nn import FlopCountAnalysis

from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
WARMUP_RUNS = 50
TIMED_RUNS = 200

def get_gpu_telemetry():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,pstate",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, encoding='utf-8').strip()
        parts = [p.strip() for p in out.split(',')]
        return {
            "gpu_name": parts[0],
            "driver_version": parts[1],
            "temperature_c": float(parts[2]),
            "utilization_pct": float(parts[3]),
            "graphics_clock_mhz": float(parts[4]),
            "memory_clock_mhz": float(parts[5]),
            "power_draw_w": float(parts[6]),
            "pstate": parts[7]
        }
    except Exception as e:
        return {"error": str(e)}

def get_accuracy_metrics():
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

def benchmark_single_configuration(model_name, res_mode, sample_img_path):
    """
    res_mode: 'native' or '640'
    """
    with open(sample_img_path, 'rb') as f:
        img_bytes = f.read()

    # Determine input dimensions
    if res_mode == '640':
        res_str = "640x640"
        img_h, img_w = 640, 640
    else:
        if 'yolo' in model_name:
            res_str = "640x640"
            img_h, img_w = 640, 640
        elif model_name == 'efficientdet':
            res_str = "512x512"
            img_h, img_w = 512, 512
        elif model_name in ['fasterrcnn', 'detr']:
            res_str = "800x1333 (native)"
            # Test image 1 (117).jpg has size 1920x911. Scaled with min_size=800, max_size=1333:
            # 1920 -> 1333, 911 -> 632
            img_h, img_w = 632, 1333

    dummy_input = torch.randn(1, 3, img_h, img_w, device=DEVICE)

    # 1. Load Model
    if 'yolo' in model_name:
        yolo = YOLO(f'checkpoints/{model_name}_best.pt')
        torch_model = yolo.model.to(DEVICE).eval()
        params_m = sum(p.numel() for p in torch_model.parameters()) / 1e6
        macs, _ = thop.profile(torch_model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9

        def model_forward():
            return torch_model(dummy_input)

        def end_to_end():
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            res = yolo.predict(arr, imgsz=img_w, rect=False, verbose=False, device=DEVICE)
            _ = res[0].boxes.xyxy
            return res

    elif model_name == 'fasterrcnn':
        torch_model = fasterrcnn_resnet50_fpn(weights=None, min_size=img_h, max_size=img_w)
        in_features = torch_model.roi_heads.box_predictor.cls_score.in_features
        torch_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
        torch_model.load_state_dict(torch.load('checkpoints/fasterrcnn_best.pt', map_location=DEVICE))
        torch_model.to(DEVICE).eval()
        params_m = sum(p.numel() for p in torch_model.parameters()) / 1e6
        macs, _ = thop.profile(torch_model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9

        def model_forward():
            return torch_model(dummy_input)

        def end_to_end():
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(arr)
            tensor = F.to_tensor(pil_img).unsqueeze(0).to(DEVICE)
            out = torch_model(tensor)[0]
            boxes = out['boxes'].cpu()
            return boxes

    elif model_name == 'efficientdet':
        config = get_efficientdet_config('tf_efficientdet_d2')
        config.num_classes = 4
        config.image_size = (img_h, img_w)
        net = EfficientDet(config, pretrained_backbone=False)
        net.load_state_dict(torch.load('checkpoints/efficientdet_best.pt', map_location=DEVICE))
        torch_model = DetBenchPredict(net).to(DEVICE).eval()
        params_m = sum(p.numel() for p in net.parameters()) / 1e6
        macs, _ = thop.profile(net, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9

        img_info = {
            "img_scale": torch.tensor([1.0], device=DEVICE),
            "img_size": torch.tensor([[img_h, img_w]], device=DEVICE)
        }
        def model_forward():
            return torch_model(dummy_input, img_info=img_info)

        def end_to_end():
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(arr, (img_w, img_h))
            norm = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            dets = torch_model(tensor, img_info=img_info)[0]
            boxes = dets[:, 0:4]
            return boxes

    elif model_name == 'detr':
        if res_mode == '640':
            processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                size={"shortest_edge": 640, "longest_edge": 640}
            )
        else:
            processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                size={"shortest_edge": 800, "longest_edge": 1333}
            )
        torch_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", num_labels=4, ignore_mismatched_sizes=True
        )
        torch_model.load_state_dict(torch.load('checkpoints/detr_best.pt', map_location=DEVICE))
        torch_model.to(DEVICE).eval()
        params_m = sum(p.numel() for p in torch_model.parameters()) / 1e6
        macs, _ = thop.profile(torch_model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9

        mask = torch.ones(1, img_h, img_w, dtype=torch.long, device=DEVICE)
        def model_forward():
            return torch_model(pixel_values=dummy_input, pixel_mask=mask)

        def end_to_end():
            arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(arr)
            inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
            outputs = torch_model(**inputs)
            target_sizes = torch.tensor([[img_h, img_w]], device=DEVICE)
            res = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)[0]
            return res

    # 2. Benchmark Model-Only
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model_forward()
            torch.cuda.synchronize()

        model_times = []
        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model_forward()
            torch.cuda.synchronize()
            model_times.append((time.perf_counter() - t0) * 1000.0)

    model_mean = float(np.mean(model_times))
    model_std = float(np.std(model_times))
    model_fps = float(1000.0 / model_mean)

    # 3. Benchmark End-to-End
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = end_to_end()
            torch.cuda.synchronize()

        e2e_times = []
        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = end_to_end()
            torch.cuda.synchronize()
            e2e_times.append((time.perf_counter() - t0) * 1000.0)

    e2e_mean = float(np.mean(e2e_times))
    e2e_std = float(np.std(e2e_times))
    e2e_fps = float(1000.0 / e2e_mean)

    print(f"[{model_name.upper()} | {res_mode} ({res_str})]")
    print(f"  Model-Only: {model_mean:.2f} ± {model_std:.2f} ms ({model_fps:.1f} FPS)")
    print(f"  End-to-End: {e2e_mean:.2f} ± {e2e_std:.2f} ms ({e2e_fps:.1f} FPS)")
    print(f"  Params: {params_m:.2f}M | GFLOPs: {gflops:.1f}")

    return {
        "model": model_name,
        "res_mode": res_mode,
        "resolution": res_str,
        "params_m": round(params_m, 2),
        "gflops": round(gflops, 1),
        "model_latency_mean_ms": round(model_mean, 2),
        "model_latency_std_ms": round(model_std, 2),
        "model_fps": round(model_fps, 1),
        "e2e_latency_mean_ms": round(e2e_mean, 2),
        "e2e_latency_std_ms": round(e2e_std, 2),
        "e2e_fps": round(e2e_fps, 1)
    }

def cross_check_fvcore():
    print("\n--- Cross-Checking GFLOPs with fvcore ---")
    results = {}
    
    # Faster R-CNN at 640x640
    frcnn_640 = fasterrcnn_resnet50_fpn(weights=None, min_size=640, max_size=640)
    frcnn_640.roi_heads.box_predictor = FastRCNNPredictor(frcnn_640.roi_heads.box_predictor.cls_score.in_features, num_classes=5)
    frcnn_640.eval()
    dummy_640 = torch.randn(1, 3, 640, 640)
    frcnn_macs_thop, _ = thop.profile(frcnn_640, inputs=(dummy_640,), verbose=False)
    frcnn_flops_fvcore = FlopCountAnalysis(frcnn_640, dummy_640)
    
    # DETR at 640x640
    detr = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=4, ignore_mismatched_sizes=True).eval()
    detr_macs_thop, _ = thop.profile(detr, inputs=(dummy_640,), verbose=False)
    mask_640 = torch.ones(1, 640, 640, dtype=torch.long)
    detr_flops_fvcore = FlopCountAnalysis(detr, (dummy_640, mask_640))

    results['fasterrcnn_640'] = {
        'thop_gflops': round(frcnn_macs_thop * 2 / 1e9, 2),
        'fvcore_macs_gflops': round(frcnn_flops_fvcore.total() / 1e9, 2),
        'fvcore_2x_gflops': round(frcnn_flops_fvcore.total() * 2 / 1e9, 2),
        'unsupported_ops': dict(frcnn_flops_fvcore.unsupported_ops())
    }
    results['detr_640'] = {
        'thop_gflops': round(detr_macs_thop * 2 / 1e9, 2),
        'fvcore_macs_gflops': round(detr_flops_fvcore.total() / 1e9, 2),
        'fvcore_2x_gflops': round(detr_flops_fvcore.total() * 2 / 1e9, 2),
        'unsupported_ops': dict(detr_flops_fvcore.unsupported_ops())
    }

    print(f"Faster R-CNN: thop={results['fasterrcnn_640']['thop_gflops']} GFLOPs, fvcore(2x)={results['fasterrcnn_640']['fvcore_2x_gflops']} GFLOPs")
    print(f"  Unsupported in fvcore: torchvision::roi_align ({results['fasterrcnn_640']['unsupported_ops'].get('torchvision::roi_align', 0)} calls)")
    print(f"DETR: thop={results['detr_640']['thop_gflops']} GFLOPs, fvcore(2x)={results['detr_640']['fvcore_2x_gflops']} GFLOPs")
    print(f"  Unsupported in fvcore: aten::scaled_dot_product_attention ({results['detr_640']['unsupported_ops'].get('aten::scaled_dot_product_attention', 0)} calls)")

    return results

def detailed_yolo_overhead_analysis(sample_img_path):
    print("\n--- Detailed YOLOv8 Overhead Analysis ---")
    with open(sample_img_path, 'rb') as f:
        img_bytes = f.read()

    # 1. Decode time
    dec_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        dec_times.append((time.perf_counter() - t0) * 1000.0)
    dec_ms = float(np.mean(dec_times))

    report = {'decode_ms': round(dec_ms, 2), 'models': {}}

    for name in ['yolov8n', 'yolov8m']:
        yolo = YOLO(f'checkpoints/{name}_best.pt')
        # Warmup
        _ = yolo.predict(arr, imgsz=640, rect=False, verbose=False)
        _ = yolo.predict(arr, imgsz=640, rect=True, verbose=False)

        # Profile rect=False (square 640x640)
        sq_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r_sq = yolo.predict(arr, imgsz=640, rect=False, verbose=False)
            _ = r_sq[0].boxes.xyxy
            torch.cuda.synchronize()
            sq_times.append((time.perf_counter() - t0) * 1000.0)

        # Profile rect=True (rectangular 320x640)
        rect_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r_rect = yolo.predict(arr, imgsz=640, rect=True, verbose=False)
            _ = r_rect[0].boxes.xyxy
            torch.cuda.synchronize()
            rect_times.append((time.perf_counter() - t0) * 1000.0)

        report['models'][name] = {
            'square_640_total_ms': round(float(np.mean(sq_times)), 2),
            'square_640_internal_speed': {k: round(v, 2) for k, v in r_sq[0].speed.items()},
            'rect_320_total_ms': round(float(np.mean(rect_times)), 2),
            'rect_320_internal_speed': {k: round(v, 2) for k, v in r_rect[0].speed.items()}
        }

    return report

def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)

    telemetry_start = get_gpu_telemetry()
    print("Initial GPU Telemetry:", telemetry_start)

    acc = get_accuracy_metrics()
    test_images = list(Path('dataset/images/test').glob('*.jpg'))
    sample_img_path = str(test_images[0])

    models = ['yolov8m', 'yolov8n', 'fasterrcnn', 'detr', 'efficientdet']
    native_benchmarks = []
    uniform_benchmarks = []

    print("\n================ RUNNING NATIVE RESOLUTION BENCHMARKS ================")
    for m in models:
        res = benchmark_single_configuration(m, 'native', sample_img_path)
        native_benchmarks.append(res)

    print("\n================ RUNNING UNIFORM 640x640 BENCHMARKS ================")
    for m in models:
        res = benchmark_single_configuration(m, '640', sample_img_path)
        uniform_benchmarks.append(res)

    telemetry_end = get_gpu_telemetry()
    print("Final GPU Telemetry:", telemetry_end)

    # Cross-checks
    fvcore_checks = cross_check_fvcore()
    yolo_overhead = detailed_yolo_overhead_analysis(sample_img_path)

    # Save comprehensive JSON and CSV
    full_output = {
        "telemetry_start": telemetry_start,
        "telemetry_end": telemetry_end,
        "native_benchmarks": native_benchmarks,
        "uniform_640_benchmarks": uniform_benchmarks,
        "fvcore_checks": fvcore_checks,
        "yolo_overhead_investigation": yolo_overhead
    }

    with open('results_v2/comprehensive_speed_benchmark.json', 'w') as f:
        json.dump(full_output, f, indent=2)
    with open('run2/comprehensive_speed_benchmark.json', 'w') as f:
        json.dump(full_output, f, indent=2)

    # Create merged DataFrame
    df_rows = []
    for m in models:
        n_res = next(x for x in native_benchmarks if x['model'] == m)
        u_res = next(x for x in uniform_benchmarks if x['model'] == m)
        m_acc = acc.get(m, {'mAP': 0, 'AP50': 0, 'AP75': 0})
        df_rows.append({
            "model": m,
            "mAP": m_acc['mAP'],
            "AP50": m_acc['AP50'],
            "AP75": m_acc['AP75'],
            "native_res": n_res['resolution'],
            "native_model_fps": n_res['model_fps'],
            "native_e2e_fps": n_res['e2e_fps'],
            "native_gflops": n_res['gflops'],
            "uniform_640_model_fps": u_res['model_fps'],
            "uniform_640_e2e_fps": u_res['e2e_fps'],
            "uniform_640_gflops": u_res['gflops'],
            "params_m": u_res['params_m']
        })

    df = pd.DataFrame(df_rows)
    df.to_csv('results_v2/speed_benchmark.csv', index=False)
    df.to_csv('run2/speed_benchmark.csv', index=False)
    print("\nMerged Benchmark Table:")
    print(df.to_string())

    # Generate Updated LaTeX Table 5
    tex = r"""\begin{table*}[htbp]
\centering
\caption{Comprehensive Speed, Accuracy, and Complexity Benchmark on African Wildlife Test Set. Accuracy metrics are evaluated at each model's native resolution. Throughput (FPS) is benchmarked under controlled conditions on an NVIDIA GeForce RTX 4050 Laptop GPU (6\,GB VRAM, Driver 610.74, P0 performance state, clock: $2130\,\text{MHz}$ graphics / $8001\,\text{MHz}$ memory, temperature: $48\text{--}52^\circ\text{C}$, power draw: $12\text{--}42\,\text{W}$) with batch size 1 and FP32 precision. Latencies averaged over 200 timed iterations following 50 warm-up runs with explicit CUDA synchronization. We report throughput at both native resolution and uniform $640 \times 640$ resolution. Best values across each metric are highlighted in \textbf{bold}.}
\label{tab:speed_benchmark}
\small
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccccc}
\toprule
 & \multicolumn{3}{c}{\textbf{Detection Accuracy (Native)}} & \textbf{Native} & \multicolumn{2}{c}{\textbf{Native Res. FPS}} & \multicolumn{2}{c}{\textbf{Uniform $640^2$ FPS}} & \textbf{Params} & \textbf{GFLOPs} \\
\cmidrule(lr){2-4} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
\textbf{Model} & \textbf{mAP@[.5:.95]} & \textbf{AP\textsubscript{50}} & \textbf{AP\textsubscript{75}} & \textbf{Resolution} & \textbf{Model\textsuperscript{\textdagger}} & \textbf{E2E\textsuperscript{\textdaggerdbl}} & \textbf{Model\textsuperscript{\textdagger}} & \textbf{E2E\textsuperscript{\textdaggerdbl}} & \textbf{(M)} & \textbf{($640^2$)\textsuperscript{*}} \\
\midrule
"""
    for _, row in df.iterrows():
        is_best_map = row['mAP'] == df['mAP'].max()
        is_best_ap50 = row['AP50'] == df['AP50'].max()
        is_best_ap75 = row['AP75'] == df['AP75'].max()
        is_best_n_mod = row['native_model_fps'] == df['native_model_fps'].max()
        is_best_n_e2e = row['native_e2e_fps'] == df['native_e2e_fps'].max()
        is_best_u_mod = row['uniform_640_model_fps'] == df['uniform_640_model_fps'].max()
        is_best_u_e2e = row['uniform_640_e2e_fps'] == df['uniform_640_e2e_fps'].max()
        is_best_params = row['params_m'] == df['params_m'].min()
        is_best_flops = row['uniform_640_gflops'] == df['uniform_640_gflops'].min()

        map_str = f"\\textbf{{{row['mAP']:.3f}}}" if is_best_map else f"{row['mAP']:.3f}"
        ap50_str = f"\\textbf{{{row['AP50']:.3f}}}" if is_best_ap50 else f"{row['AP50']:.3f}"
        ap75_str = f"\\textbf{{{row['AP75']:.3f}}}" if is_best_ap75 else f"{row['AP75']:.3f}"
        n_mod_str = f"\\textbf{{{row['native_model_fps']:.1f}}}" if is_best_n_mod else f"{row['native_model_fps']:.1f}"
        n_e2e_str = f"\\textbf{{{row['native_e2e_fps']:.1f}}}" if is_best_n_e2e else f"{row['native_e2e_fps']:.1f}"
        u_mod_str = f"\\textbf{{{row['uniform_640_model_fps']:.1f}}}" if is_best_u_mod else f"{row['uniform_640_model_fps']:.1f}"
        u_e2e_str = f"\\textbf{{{row['uniform_640_e2e_fps']:.1f}}}" if is_best_u_e2e else f"{row['uniform_640_e2e_fps']:.1f}"
        params_str = f"\\textbf{{{row['params_m']:.2f}}}" if is_best_params else f"{row['params_m']:.2f}"
        flops_str = f"\\textbf{{{row['uniform_640_gflops']:.1f}}}" if is_best_flops else f"{row['uniform_640_gflops']:.1f}"

        # Clean display model name
        m_display = {
            'yolov8m': 'YOLOv8m',
            'yolov8n': 'YOLOv8n',
            'fasterrcnn': 'Faster R-CNN',
            'detr': 'DETR',
            'efficientdet': 'EfficientDet-D2'
        }.get(row['model'], row['model'])

        res_clean = row['native_res'].replace(' (native)', '')

        tex += f"{m_display} & {map_str} & {ap50_str} & {ap75_str} & {res_clean} & {n_mod_str} & {n_e2e_str} & {u_mod_str} & {u_e2e_str} & {params_str} & {flops_str} \\\\\n"

    tex += r"""\bottomrule
\multicolumn{11}{l}{\footnotesize \textsuperscript{\textdagger} Model-only forward pass throughput. \textsuperscript{\textdaggerdbl} End-to-end throughput including image decode, preprocessing (resizing/normalization), and postprocessing (NMS/Hungarian).} \\
\multicolumn{11}{l}{\footnotesize \textsuperscript{*} GFLOPs computed via standard convention ($\text{FLOPs} = 2 \times \text{MACs}$) at $640 \times 640$. Cross-checked with \texttt{fvcore}; see text for attention and RoIAlign accounting.} \\
\end{tabular}
\end{table*}
"""
    with open('run2/table5_speed_accuracy.tex', 'w') as f:
        f.write(tex)
    with open('results_v2/table5_speed_accuracy.tex', 'w') as f:
        f.write(tex)
    print("\nSaved updated LaTeX Table 5 to run2/table5_speed_accuracy.tex and results_v2/table5_speed_accuracy.tex")

if __name__ == '__main__':
    main()
