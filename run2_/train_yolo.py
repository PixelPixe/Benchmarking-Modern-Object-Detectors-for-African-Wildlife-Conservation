#!/usr/bin/env python3
"""
run2/train_yolo.py - Seed-controlled YOLOv8 training script reading configs/common.yaml
Outputs to checkpoints_v2/ and logs_v2/
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings

from utils import set_seeds, save_resolved_hyperparams, setup_environment, load_common_config, DEVICE

def train_yolo(model_name, seed=42, epochs=None):
    cfg = load_common_config()
    common_cfg = cfg['common']
    model_cfg = cfg['models'][model_name]
    aug_cfg = cfg['augmentation']

    set_seeds(seed)
    print(f"\n--- Training {model_name.upper()} (Seed: {seed}) ---")

    checkpoints_base = os.path.abspath('checkpoints_v2')
    settings.update({'runs_dir': checkpoints_base})

    actual_epochs = epochs if epochs is not None else common_cfg['epochs']

    params = {
        'epochs': actual_epochs,
        'imgsz': common_cfg['image_size'],
        'batch': model_cfg['batch_size'],
        'optimizer': model_cfg['optimizer'],
        'lr0': model_cfg['lr0'],
        'lrf': model_cfg['lrf'],
        'weight_decay': model_cfg['weight_decay'],
        'cos_lr': True if model_cfg['schedule'] == 'cosine' else False,
        'warmup_epochs': model_cfg['warmup_epochs'],
        'seed': seed,
        'mosaic': aug_cfg['yolo_specific']['mosaic'],
        'mixup': aug_cfg['yolo_specific']['mixup'],
        'flipud': aug_cfg['vertical_flip_prob'],
        'fliplr': aug_cfg['horizontal_flip_prob'],
        'hsv_h': aug_cfg['yolo_specific']['hsv_h'],
        'hsv_s': aug_cfg['yolo_specific']['hsv_s'],
        'hsv_v': aug_cfg['yolo_specific']['hsv_v'],
        'device': 0,
        'amp': common_cfg['amp'],
        'cache': False,
        'workers': common_cfg['num_workers'],
    }

    # Save resolved hyperparameters
    save_resolved_hyperparams(model_name, seed, params)

    run_name = f"{model_name}_seed{seed}"
    run_dir = os.path.join(checkpoints_base, run_name)
    best_weights = os.path.join(run_dir, 'weights', 'best.pt')
    last_weights = os.path.join(run_dir, 'weights', 'last.pt')
    dest_best = os.path.join(checkpoints_base, f"{model_name}_seed{seed}_best.pt")

    if os.path.exists(best_weights):
        print(f"[{model_name}] Training already complete. Checkpoint exists at {best_weights}")
    elif os.path.exists(last_weights):
        print(f"[{model_name}] Resuming from {last_weights}")
        model = YOLO(last_weights)
        model.train(resume=True)
    else:
        print(f"[{model_name}] Starting fresh training from {model_cfg['weights']}")
        model = YOLO(model_cfg['weights'])
        model.train(
            data='dataset.yaml',
            project=checkpoints_base,
            name=run_name,
            exist_ok=True,
            **params
        )

    if os.path.exists(best_weights):
        shutil.copy(best_weights, dest_best)
        print(f"[{model_name}] Best weights saved to {dest_best}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with fair harmonized settings")
    parser.add_argument('--model', type=str, default='yolov8n', choices=['yolov8n', 'yolov8m'], help='Model variant')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=None, help='Override epoch count')
    args = parser.parse_args()

    setup_environment()
    train_yolo(args.model, seed=args.seed, epochs=args.epochs)

if __name__ == '__main__':
    main()
