#!/usr/bin/env python3
"""
run2/generate_qualitative_figures.py - Comparative Prediction Grids for WildCareAI2

Rewritten per peer-review feedback:
1. Strict descriptive captions: describes only visible bounding boxes, classes, and scores, with zero causal claims.
2. Explicit image selection criteria documented for each figure.
3. Added a dedicated high-confidence failure case (Figure 4) for the best model (YOLOv8m).
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']
CLASS_COLORS = {
    0: '#2563EB',  # Buffalo: Blue
    1: '#059669',  # Elephant: Green
    2: '#7C3AED',  # Rhino: Purple
    3: '#D97706',  # Zebra: Amber
}
GT_COLOR = '#DC2626'  # Red for Ground Truth

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)

def draw_boxes(ax, image_rgb, boxes, title, is_gt=False, score_thresh=0.25):
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.axis('off')

    for b in boxes:
        if is_gt:
            cls_id = b['category_id']
            x, y, bw, bh = b['bbox']
            label = f"GT: {CLASSES[cls_id]}"
            color = CLASS_COLORS.get(cls_id, GT_COLOR)
            alpha = 0.95
            lw = 2.2
        else:
            cls_id = b['category_id']
            score = b.get('score', 0.0)
            if score < score_thresh:
                continue
            x, y, bw, bh = b['bbox']
            label = f"{CLASSES[cls_id]} {score:.2f}"
            color = CLASS_COLORS.get(cls_id, '#3B82F6')
            alpha = 0.9
            lw = 2.0

        rect = patches.Rectangle(
            (x, y), bw, bh, linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha
        )
        ax.add_patch(rect)

        bbox_props = dict(boxstyle="square,pad=0.15", fc=color, ec="none", alpha=0.85)
        ax.text(
            x, max(0, y - 3), label,
            color='white', fontsize=7.5, fontweight='bold',
            bbox=bbox_props, verticalalignment='bottom'
        )

def create_comparison_grid(img_id, coco_gt, all_preds, output_path, suptitle_text=""):
    img_info = coco_gt['images'][img_id]
    img_name = img_info['file_name']
    img_path = Path('dataset/images/test') / img_name
    if not img_path.exists():
        print(f"Warning: Image {img_path} not found.")
        return

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt_boxes = [ann for ann in coco_gt['annotations'] if ann['image_id'] == img_id]

    model_order = [
        ('Ground Truth', None, True),
        ('YOLOv8m', all_preds['yolov8m'].get(img_id, []), False),
        ('YOLOv8n', all_preds['yolov8n'].get(img_id, []), False),
        ('Faster R-CNN', all_preds['fasterrcnn'].get(img_id, []), False),
        ('DETR', all_preds['detr'].get(img_id, []), False),
        ('EfficientDet-D2', all_preds['efficientdet'].get(img_id, []), False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    for idx, (title, preds, is_gt) in enumerate(model_order):
        ax = axes[idx]
        if is_gt:
            draw_boxes(ax, img_rgb, gt_boxes, f"(a) {title}", is_gt=True)
        else:
            draw_boxes(ax, img_rgb, preds, f"({chr(98+idx-1)}) {title}", is_gt=False, score_thresh=0.25)

    plt.suptitle(
        f"Test Image: {img_name} (ID: {img_id}) — {suptitle_text}",
        fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved qualitative grid: {output_path}")

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    gt_path = 'coco_annotations/instances_test.json'
    if not os.path.exists(gt_path):
        gt_path = 'dataset/annotations/instances_test.json'
    coco_gt = load_json(gt_path)

    model_keys = ['yolov8m', 'yolov8n', 'fasterrcnn', 'detr', 'efficientdet']
    all_preds = {}
    for k in model_keys:
        p_path = f'results_v2/predictions_{k}.json'
        p_list = load_json(p_path)
        d = {}
        for item in p_list:
            d.setdefault(item['image_id'], []).append(item)
        all_preds[k] = d

    out_dir = Path('results_v2/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Figure 1: Clean Multi-Animal Detection
    # Selection: First test-set image containing multiple unoccluded instances of Buffalo.
    create_comparison_grid(
        img_id=0,
        coco_gt=coco_gt,
        all_preds=all_preds,
        output_path=out_dir / 'fig1_success_case.png',
        suptitle_text="Multi-Instance African Buffalo Herd (Test Image 0: 3 Annotated Buffaloes)"
    )

    # 2. Figure 2: Small-Scale Instance
    # Selection: One of only three test-set images containing an instance in the COCO small scale bin (<1024 px^2).
    create_comparison_grid(
        img_id=102,
        coco_gt=coco_gt,
        all_preds=all_preds,
        output_path=out_dir / 'fig2_small_object_rhino.png',
        suptitle_text="Small-Scale Rhino Instance (Test Image 102: 1 Small Rhino, 1 Medium Rhino)"
    )

    # 3. Figure 3: Dense Clustered Herd
    # Selection: Test-set image with the highest instance density (10 zebra annotations).
    create_comparison_grid(
        img_id=60,
        coco_gt=coco_gt,
        all_preds=all_preds,
        output_path=out_dir / 'fig3_zebra_stripe_confusion.png',
        suptitle_text="Dense Zebra Cluster (Test Image 60: 10 Annotated Zebras)"
    )

    # 4. Figure 4: Failure Case for Best Model (YOLOv8m)
    # Selection: High-confidence false positive error produced by the top-performing model, YOLOv8m.
    create_comparison_grid(
        img_id=116,
        coco_gt=coco_gt,
        all_preds=all_preds,
        output_path=out_dir / 'fig4_yolov8m_failure_case.png',
        suptitle_text="YOLOv8m Detection Without Matching Annotation on Open Terrain (Test Image 116)"
    )

    print("\nAll 4 qualitative figures generated successfully.")

if __name__ == '__main__':
    main()
