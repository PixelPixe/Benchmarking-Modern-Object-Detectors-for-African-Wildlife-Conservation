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
WARMUP_RUNS = 20

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    test_imgs = sorted(list(Path('dataset/images/test').glob('*.jpg')))
    raw_imgs = [(p.name, open(p, 'rb').read()) for p in test_imgs]
    n_images = len(raw_imgs)
    print(f"Loaded {n_images} test images into memory for uniform 640x640 benchmark.")

    # 1. Load models
    print("Loading models...")
    yolo_m = YOLO('checkpoints/yolov8m_best.pt')
    yolo_m_torch = yolo_m.model.to(DEVICE).eval()

    yolo_n = YOLO('checkpoints/yolov8n_best.pt')
    yolo_n_torch = yolo_n.model.to(DEVICE).eval()

    frcnn = fasterrcnn_resnet50_fpn(weights=None, min_size=640, max_size=640)
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(frcnn.roi_heads.box_predictor.cls_score.in_features, num_classes=5)
    frcnn.load_state_dict(torch.load('checkpoints/fasterrcnn_best.pt', map_location=DEVICE))
    frcnn.to(DEVICE).eval()

    detr_proc = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', size={'shortest_edge': 640, 'longest_edge': 640})
    detr = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=4, ignore_mismatched_sizes=True)
    detr.load_state_dict(torch.load('checkpoints/detr_best.pt', map_location=DEVICE))
    detr.to(DEVICE).eval()

    cfg = get_efficientdet_config('tf_efficientdet_d2')
    cfg.num_classes = 4
    cfg.image_size = (640, 640)
    net = EfficientDet(cfg, pretrained_backbone=False)
    net.load_state_dict(torch.load('checkpoints/efficientdet_best.pt', map_location=DEVICE))
    effdet = DetBenchPredict(net).to(DEVICE).eval()

    # Pre-allocate dummy inputs for model forward profiling at 640x640
    dummy_640 = torch.randn(1, 3, 640, 640, device=DEVICE)

    # 2. Warmup
    print("Warming up models on GPU...")
    for _ in range(WARMUP_RUNS):
        _ = yolo_m_torch(dummy_640)
        _ = yolo_n_torch(dummy_640)
        _ = frcnn(dummy_640)
        _ = detr(dummy_640)
        _ = effdet(dummy_640, img_info={'img_scale': torch.tensor([1.0], device=DEVICE), 'img_size': torch.tensor([[640, 640]], device=DEVICE)})
        torch.cuda.synchronize()

    results = []

    # Model profiling function
    def benchmark_model(model_name, forward_fn, e2e_fn):
        print(f"Benchmarking {model_name} at uniform 640x640 over {n_images} images...")
        # Model forward
        forward_times = []
        for _ in range(n_images):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = forward_fn()
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - t0) * 1000.0)

        # End to end
        e2e_times = []
        for name, b in raw_imgs:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = e2e_fn(b)
            torch.cuda.synchronize()
            e2e_times.append((time.perf_counter() - t0) * 1000.0)

        f_med = float(np.median(forward_times))
        f_iqr = float(np.percentile(forward_times, 75) - np.percentile(forward_times, 25))
        f_fps = float(1000.0 / f_med) if f_med > 0 else 0.0

        e_med = float(np.median(e2e_times))
        e_iqr = float(np.percentile(e2e_times, 75) - np.percentile(e2e_times, 25))
        e_fps = float(1000.0 / e_med) if e_med > 0 else 0.0

        print(f"  Forward: {f_med:.2f} (IQR {f_iqr:.2f}) ms -> {f_fps:.1f} FPS")
        print(f"  E2E:     {e_med:.2f} (IQR {e_iqr:.2f}) ms -> {e_fps:.1f} FPS")

        return {
            'model': model_name,
            'resolution': '640x640',
            'forward_median_ms': round(f_med, 1),
            'forward_iqr_ms': round(f_iqr, 1),
            'forward_fps': round(f_fps, 1),
            'e2e_median_ms': round(e_med, 1),
            'e2e_iqr_ms': round(e_iqr, 1),
            'e2e_fps': round(e_fps, 1),
            'forward_mean_ms': round(float(np.mean(forward_times)), 2),
            'forward_std_ms': round(float(np.std(forward_times)), 2),
            'e2e_mean_ms': round(float(np.mean(e2e_times)), 2),
            'e2e_std_ms': round(float(np.std(e2e_times)), 2)
        }

    # YOLOv8m
    def e2e_ym(b):
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        res = yolo_m.predict(arr, imgsz=640, rect=False, verbose=False, device=DEVICE)
        _ = res[0].boxes.xyxy
        return res
    results.append(benchmark_model('yolov8m', lambda: yolo_m_torch(dummy_640), e2e_ym))

    # YOLOv8n
    def e2e_yn(b):
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        res = yolo_n.predict(arr, imgsz=640, rect=False, verbose=False, device=DEVICE)
        _ = res[0].boxes.xyxy
        return res
    results.append(benchmark_model('yolov8n', lambda: yolo_n_torch(dummy_640), e2e_yn))

    # Faster R-CNN
    def e2e_fr(b):
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        t = F.to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
        out = frcnn(t)[0]
        _ = out['boxes']
        return out
    results.append(benchmark_model('fasterrcnn', lambda: frcnn(dummy_640), e2e_fr))

    # DETR
    def e2e_detr(b):
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        inputs = detr_proc(images=Image.fromarray(rgb), return_tensors='pt').to(DEVICE)
        out = detr(**inputs)
        _ = out.logits
        return out
    results.append(benchmark_model('detr', lambda: detr(dummy_640), e2e_detr))

    # EfficientDet-D2
    def e2e_effdet(b):
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        res = cv2.resize(rgb, (640, 640))
        norm = (res / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        out = effdet(t, img_info={'img_scale': torch.tensor([1.0], device=DEVICE), 'img_size': torch.tensor([[640, 640]], device=DEVICE)})
        return out
    results.append(benchmark_model('efficientdet', lambda: effdet(dummy_640, img_info={'img_scale': torch.tensor([1.0], device=DEVICE), 'img_size': torch.tensor([[640, 640]], device=DEVICE)}), e2e_effdet))

    df = pd.DataFrame(results)
    df.to_csv('results_v2/dataset_speed_uniform_640.csv', index=False)
    df.to_csv('run2/dataset_speed_uniform_640.csv', index=False)
    with open('results_v2/dataset_speed_uniform_640.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark completed and saved to results_v2/dataset_speed_uniform_640.csv")

if __name__ == '__main__':
    main()
