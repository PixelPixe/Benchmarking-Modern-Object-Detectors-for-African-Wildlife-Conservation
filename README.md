# Benchmarking Modern Object Detectors for African Wildlife Conservation

A comparison of five object detectors on a four-class African wildlife dataset
(buffalo, elephant, rhinoceros, zebra):

- YOLOv8m and YOLOv8n (one-stage, anchor-free)
- Faster R-CNN, ResNet-50 FPN (two-stage)
- EfficientDet-D2 (one-stage, compound-scaled)
- DETR, ResNet-50 (transformer)

Paper: [under review / add link when available]

## Main results

Evaluated on a 145-image held-out test set with standard COCO metrics.

| Model | mAP@[.5:.95] | AP50 | AP75 |
|---|---|---|---|
| YOLOv8m | 0.777 | 0.904 | 0.848 |
| YOLOv8n | 0.744 | 0.886 | 0.816 |
| Faster R-CNN | 0.702 | 0.901 | 0.830 |
| DETR | 0.675 | 0.858 | 0.743 |
| EfficientDet-D2 | 0.597 | 0.849 | 0.686 |

Speed on an NVIDIA RTX 4050 Laptop GPU (batch size 1, FP32, native input resolution):

| Model | Model-only FPS | End-to-end FPS |
|---|---|---|
| YOLOv8n | 117.5 | 62.5 |
| YOLOv8m | 70.0 | 46.0 |
| EfficientDet-D2 | 20.1 | 15.6 |
| DETR | 15.7 | 7.4 |
| Faster R-CNN | 10.9 | 10.2 |

**Important caveats**
- Each model was trained once (100 epochs), so differences of one to three mAP points
  may fall within seed-to-seed variability.
- Training recipes differ between models (batch size, optimizer, augmentation, resolution,
  mixed precision) and may favour the YOLOv8 models.
- The test set contains only three small objects, so small-object AP is not meaningful.
- Bootstrap confidence intervals cover test-set variability only, not training variability.

## Dataset

[Ultralytics African Wildlife dataset](https://docs.ultralytics.com/datasets/detect/african-wildlife/).
The raw data had 1,504 images. After removing images with identical perceptual hashes,
1,441 images were split 80/10/10 (1,152 train / 144 validation / 145 test).
The images are not included in this repository.

## Repository layout

- `run2/` contains the Python scripts:
  - `train_yolo.py`, `train_fasterrcnn.py`, `train_efficientdet.py`, `train_detr.py`: training
  - `evaluate_all.py`: COCO evaluation of all models
  - `benchmark_*.py`, `run_interleaved_benchmark.py`: speed measurements
  - `bootstrap_ci.py`, `bootstrap_high_iou.py`: paired bootstrap significance tests
  - `audit_duplicates.py`, `analyze_duplicates_pixels.py`, `compute_hamming_distances.py`: duplicate-image checks
  - `dataset_stats.py`: instance counts by class and object size
  - `generate_qualitative_figures.py`: detection figures
- `results_v2/` contains the outputs: metrics, per-model predictions, speed measurements,
  bootstrap results, duplicate audits, LaTeX tables and figures.
  `results_v2/archive/` holds superseded speed measurements that are not used in the paper.

## Environment

Python 3.11, PyTorch 2.6.0, CUDA 12.4, TorchVision 0.21.0, Ultralytics 8.4.23,
Hugging Face Transformers 4.49.0, pycocotools 2.0.8.


