#!/usr/bin/env python3
"""
run2/train_fasterrcnn.py - Seed-controlled Faster R-CNN training script reading configs/common.yaml
Features: 640x640 uniform resolution, AMP enabled, horizontal flip + color jitter augmentations, resume support.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T
import torchvision.transforms.functional as F
from pycocotools.coco import COCO

from utils import set_seeds, save_resolved_hyperparams, setup_environment, load_common_config, DEVICE

class AnimalDataset(Dataset):
    def __init__(self, split, is_train=False, aug_cfg=None):
        self.root = f"dataset/images/{split}"
        self.coco = COCO(f"coco_annotations/instances_{split}.json")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.is_train = is_train
        self.aug_cfg = aug_cfg

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        w_orig, h_orig = img.size

        boxes = []
        labels = []
        for ann in coco_annotation:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] + 1)  # 1-indexed (0 is background)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        # Training Augmentations
        if self.is_train and self.aug_cfg:
            # 1. Random Horizontal Flip
            if torch.rand(1).item() < self.aug_cfg.get('horizontal_flip_prob', 0.5):
                img = F.hflip(img)
                if len(boxes) > 0:
                    boxes[:, [0, 2]] = w_orig - boxes[:, [2, 0]]
            
            # 2. Color Jitter
            cj = self.aug_cfg.get('color_jitter', {})
            if cj:
                jitter = T.ColorJitter(
                    brightness=cj.get('brightness', 0.2),
                    contrast=cj.get('contrast', 0.2),
                    saturation=cj.get('saturation', 0.2)
                )
                img = jitter(img)

        img_tensor = F.to_tensor(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id], dtype=torch.int64)
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_faster_rcnn(seed=42, epochs=None):
    cfg = load_common_config()
    common_cfg = cfg['common']
    model_cfg = cfg['models']['fasterrcnn']
    aug_cfg = cfg['augmentation']

    set_seeds(seed)
    print(f"\n--- Training Faster R-CNN (Seed: {seed}) ---")

    actual_epochs = epochs if epochs is not None else common_cfg['epochs']
    batch_size = model_cfg['batch_size']
    img_size = common_cfg['image_size']

    resolved_params = {
        'model': 'fasterrcnn',
        'seed': seed,
        'epochs': actual_epochs,
        'batch_size': batch_size,
        'optimizer': model_cfg['optimizer'],
        'lr': model_cfg['lr'],
        'momentum': model_cfg['momentum'],
        'weight_decay': model_cfg['weight_decay'],
        'schedule': model_cfg['schedule'],
        'step_size': model_cfg['step_size'],
        'gamma': model_cfg['gamma'],
        'image_size': img_size,
        'amp': common_cfg['amp'],
        'device': str(DEVICE)
    }
    save_resolved_hyperparams('fasterrcnn', seed, resolved_params)

    # Model definition with exact 640x640 resolution
    weights = getattr(FasterRCNN_ResNet50_FPN_Weights, 'COCO_V1')
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=img_size, max_size=img_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator
    model.to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=model_cfg['lr'], momentum=model_cfg['momentum'], weight_decay=model_cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=model_cfg['step_size'], gamma=model_cfg['gamma'])
    scaler = torch.cuda.amp.GradScaler(enabled=common_cfg['amp'])

    checkpoints_base = Path('checkpoints_v2')
    checkpoints_base.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoints_base / f'fasterrcnn_seed{seed}_best.pt'
    last_ckpt = checkpoints_base / f'fasterrcnn_seed{seed}_last.pt'

    start_epoch = 0
    best_loss = float('inf')

    if last_ckpt.exists():
        print(f"[Faster R-CNN] Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
    elif best_ckpt.exists():
        print(f"[Faster R-CNN] Training already completed. Best checkpoint exists at {best_ckpt}")
        return

    train_dataset = AnimalDataset(split='train', is_train=True, aug_cfg=aug_cfg)
    val_dataset   = AnimalDataset(split='val', is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=common_cfg['num_workers'], collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=common_cfg['num_workers'], collate_fn=collate_fn
    )

    for epoch in range(start_epoch, actual_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=common_cfg['amp']):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                with torch.cuda.amp.autocast(enabled=common_cfg['amp']):
                    loss_dict = model(images, targets)
                    val_loss += sum(loss_dict.values()).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:>3}/{actual_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
        }, last_ckpt)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> Best model saved to {best_ckpt} (val_loss: {best_loss:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    setup_environment()
    train_faster_rcnn(seed=args.seed, epochs=args.epochs)

if __name__ == '__main__':
    main()
