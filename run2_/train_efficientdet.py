#!/usr/bin/env python3
"""
run2/train_efficientdet.py - Seed-controlled EfficientDet-D2 training script reading configs/common.yaml
Features: 640x640 uniform resolution, AMP enabled, Albumentations horizontal flip + color jitter, resume support.
"""

import os
import cv2
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from utils import set_seeds, save_resolved_hyperparams, setup_environment, load_common_config, DEVICE

def get_transforms(img_size, is_train=True, aug_cfg=None):
    if is_train and aug_cfg:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=aug_cfg.get('horizontal_flip_prob', 0.5)),
            A.ColorJitter(
                brightness=aug_cfg.get('color_jitter', {}).get('brightness', 0.2),
                contrast=aug_cfg.get('color_jitter', {}).get('contrast', 0.2),
                saturation=aug_cfg.get('color_jitter', {}).get('saturation', 0.2),
                p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], clip=True))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], clip=True))

class EffDetDataset(Dataset):
    def __init__(self, split, img_size=640, is_train=False, aug_cfg=None):
        self.root = f"dataset/images/{split}"
        self.coco = COCO(f"coco_annotations/instances_{split}.json")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = get_transforms(img_size, is_train=is_train, aug_cfg=aug_cfg)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        boxes, labels = [], []
        for ann in coco_annotation:
            x, y, w, h = ann['bbox']
            x = max(0.0, float(x))
            y = max(0.0, float(y))
            w = min(float(w), img_w - x)
            h = min(float(h), img_h - y)
            if w <= 1.0 or h <= 1.0:
                continue
            boxes.append([x, y, w, h])
            labels.append(ann['category_id'] + 1)  # 1-indexed (0 is background)

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        effdet_boxes = []
        for b in boxes:
            x_min, y_min, w, h = b
            effdet_boxes.append([y_min, x_min, y_min + h, x_min + w])

        target = {
            'bbox': torch.tensor(effdet_boxes, dtype=torch.float32),
            'cls': torch.tensor(labels, dtype=torch.int64)
        }
        return image, target

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    max_boxes = max([len(t['bbox']) for t in targets])
    max_boxes = max(max_boxes, 1)

    batch_bboxes = torch.zeros((len(batch), max_boxes, 4), dtype=torch.float32)
    batch_cls = torch.zeros((len(batch), max_boxes), dtype=torch.int64)

    for i, t in enumerate(targets):
        num_boxes = len(t['bbox'])
        if num_boxes > 0:
            batch_bboxes[i, :num_boxes] = t['bbox']
            batch_cls[i, :num_boxes] = t['cls']

    return images, {'bbox': batch_bboxes, 'cls': batch_cls}

def train_efficientdet(seed=42, epochs=None):
    cfg = load_common_config()
    common_cfg = cfg['common']
    model_cfg = cfg['models']['efficientdet']
    aug_cfg = cfg['augmentation']

    set_seeds(seed)
    print(f"\n--- Training EfficientDet-D2 (Seed: {seed}) ---")

    actual_epochs = epochs if epochs is not None else common_cfg['epochs']
    batch_size = model_cfg['batch_size']
    img_size = common_cfg['image_size']

    resolved_params = {
        'model': 'efficientdet',
        'seed': seed,
        'epochs': actual_epochs,
        'batch_size': batch_size,
        'optimizer': model_cfg['optimizer'],
        'lr': model_cfg['lr'],
        'weight_decay': model_cfg['weight_decay'],
        'schedule': model_cfg['schedule'],
        'T_max': model_cfg['T_max'],
        'eta_min': model_cfg['eta_min'],
        'image_size': img_size,
        'amp': common_cfg['amp'],
        'device': str(DEVICE)
    }
    save_resolved_hyperparams('efficientdet', seed, resolved_params)

    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = 4
    config.image_size = (img_size, img_size)

    net = EfficientDet(config, pretrained_backbone=True)
    model = DetBenchTrain(net, config)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg['lr'], weight_decay=model_cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_epochs, eta_min=model_cfg['eta_min'])
    scaler = torch.cuda.amp.GradScaler(enabled=common_cfg['amp'])

    checkpoints_base = Path('checkpoints_v2')
    checkpoints_base.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoints_base / f'efficientdet_seed{seed}_best.pt'
    last_ckpt = checkpoints_base / f'efficientdet_seed{seed}_last.pt'

    start_epoch = 0
    best_loss = float('inf')

    if last_ckpt.exists():
        print(f"[EfficientDet] Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
    elif best_ckpt.exists():
        print(f"[EfficientDet] Training already complete. Best checkpoint exists at {best_ckpt}")
        return

    train_dataset = EffDetDataset(split='train', img_size=img_size, is_train=True, aug_cfg=aug_cfg)
    val_dataset   = EffDetDataset(split='val', img_size=img_size, is_train=False)

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
            images = images.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=common_cfg['amp']):
                loss_dict = model(images, targets)
                loss = loss_dict['loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_cfg['grad_clip_norm'])
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in targets.items()}
                with torch.cuda.amp.autocast(enabled=common_cfg['amp']):
                    loss_dict = model(images, targets)
                    val_loss += loss_dict['loss'].item()

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
            torch.save(net.state_dict(), best_ckpt)
            print(f"  -> Best model saved to {best_ckpt} (val_loss: {best_loss:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    setup_environment()
    train_efficientdet(seed=args.seed, epochs=args.epochs)

if __name__ == '__main__':
    main()
