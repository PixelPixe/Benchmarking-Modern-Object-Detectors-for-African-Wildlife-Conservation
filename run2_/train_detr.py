#!/usr/bin/env python3
"""
run2/train_detr.py - Seed-controlled DETR training script reading configs/common.yaml
Features: 640x640 resolution override, AMP enabled, support for 100 vs 300 epoch schedules, resume support.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
import torchvision.transforms as T
import torchvision.transforms.functional as F
from pycocotools.coco import COCO

from utils import set_seeds, save_resolved_hyperparams, setup_environment, load_common_config, DEVICE

class DetrDataset(Dataset):
    def __init__(self, split, is_train=False, aug_cfg=None):
        self.root = f"dataset/images/{split}"
        self.coco = COCO(f"coco_annotations/instances_{split}.json")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.is_train = is_train
        self.aug_cfg = aug_cfg

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        w_orig, h_orig = image.size

        # Simple Augmentation for DETR
        if self.is_train and self.aug_cfg:
            cj = self.aug_cfg.get('color_jitter', {})
            if cj:
                jitter = T.ColorJitter(
                    brightness=cj.get('brightness', 0.2),
                    contrast=cj.get('contrast', 0.2),
                    saturation=cj.get('saturation', 0.2)
                )
                image = jitter(image)

        annotations = []
        for ann in coco_annotation:
            if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                continue
            annotations.append({
                "area": ann['area'],
                "iscrowd": ann.get('iscrowd', 0),
                "image_id": img_id,
                "bbox": ann['bbox'],
                "category_id": ann['category_id']
            })

        target = {'image_id': img_id, 'annotations': annotations}
        return image, target

def train_detr(seed=42, epochs=None):
    cfg = load_common_config()
    common_cfg = cfg['common']
    model_cfg = cfg['models']['detr']
    aug_cfg = cfg['augmentation']

    set_seeds(seed)
    actual_epochs = epochs if epochs is not None else common_cfg['epochs']
    img_size = common_cfg['image_size']
    batch_size = model_cfg['batch_size']

    print(f"\n--- Training DETR ResNet-50 (Seed: {seed}, Epochs: {actual_epochs}) ---")

    # Determine step size based on epoch count
    if actual_epochs > 150:
        step_size = model_cfg.get('extended_step_size', 200)
    else:
        step_size = model_cfg.get('step_size', 70)

    resolved_params = {
        'model': 'detr',
        'seed': seed,
        'epochs': actual_epochs,
        'batch_size': batch_size,
        'optimizer': model_cfg['optimizer'],
        'backbone_lr': model_cfg['backbone_lr'],
        'transformer_lr': model_cfg['transformer_lr'],
        'weight_decay': model_cfg['weight_decay'],
        'schedule': model_cfg['schedule'],
        'step_size': step_size,
        'gamma': model_cfg['gamma'],
        'image_size': img_size,
        'amp': common_cfg['amp'],
        'weights': model_cfg['weights'],
        'device': str(DEVICE)
    }
    save_resolved_hyperparams('detr', seed, resolved_params)

    # Initialize Processor with 640x640 resolution
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": img_size, "longest_edge": img_size}
    )

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        encoding = processor(images=pixel_values, annotations=targets, return_tensors="pt")
        return encoding

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': model.model.backbone.parameters(), 'lr': model_cfg['backbone_lr']},
        {'params': model.model.encoder.parameters(),  'lr': model_cfg['transformer_lr']},
        {'params': model.model.decoder.parameters(),  'lr': model_cfg['transformer_lr']},
        {'params': model.class_labels_classifier.parameters(), 'lr': model_cfg['transformer_lr']},
        {'params': model.bbox_predictor.parameters(), 'lr': model_cfg['transformer_lr']},
    ], weight_decay=model_cfg['weight_decay'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=model_cfg['gamma'])
    scaler = torch.cuda.amp.GradScaler(enabled=common_cfg['amp'])

    checkpoints_base = Path('checkpoints_v2')
    checkpoints_base.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoints_base / f'detr_seed{seed}_best.pt'
    last_ckpt = checkpoints_base / f'detr_seed{seed}_last.pt'

    start_epoch = 0
    best_loss = float('inf')

    if last_ckpt.exists():
        print(f"[DETR] Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
    elif best_ckpt.exists():
        print(f"[DETR] Training already complete. Best checkpoint exists at {best_ckpt}")
        return

    train_dataset = DetrDataset(split='train', is_train=True, aug_cfg=aug_cfg)
    val_dataset   = DetrDataset(split='val', is_train=False)

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

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(DEVICE)
            pixel_mask = batch["pixel_mask"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=common_cfg['amp']):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_cfg['grad_clip_norm'])
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                pixel_mask = batch["pixel_mask"].to(DEVICE)
                labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
                with torch.cuda.amp.autocast(enabled=common_cfg['amp']):
                    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                    val_loss += outputs.loss.item()

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
    train_detr(seed=args.seed, epochs=args.epochs)

if __name__ == '__main__':
    main()
