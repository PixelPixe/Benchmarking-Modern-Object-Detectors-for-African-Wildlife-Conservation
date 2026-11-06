#!/usr/bin/env python3
"""
run2/bootstrap_ci.py - Paired Bootstrap Confidence Intervals for WildCareAI2

Performs 1,000 paired bootstrap iterations over the 145 test images with replacement
comparing the top models:
1. YOLOv8m vs. Faster R-CNN
2. YOLOv8m vs. YOLOv8n

Evaluates:
- mAP@[.5:.95]
- AP_50
- AP_75
Computes 95% bootstrap confidence intervals (2.5th and 97.5th percentiles)
and two-sided empirical p-values testing H0: Delta = 0.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_predictions(pred_path):
    with open(pred_path, 'r') as f:
        return json.load(f)

def run_paired_bootstrap(num_bootstraps=1000, seed=42):
    np.random.seed(seed)
    
    gt_path = 'coco_annotations/instances_test.json'
    if not os.path.exists(gt_path):
        gt_path = 'dataset/annotations/instances_test.json'
    
    with open(gt_path, 'r') as f:
        coco_gt_dict = json.load(f)

    img_ids = [img['id'] for img in coco_gt_dict['images']]
    n_images = len(img_ids)
    print(f"Total test images for bootstrap: {n_images}")

    # Load predictions
    preds_yolov8m = load_predictions('results_v2/predictions_yolov8m.json')
    preds_fasterrcnn = load_predictions('results_v2/predictions_fasterrcnn.json')
    preds_yolov8n = load_predictions('results_v2/predictions_yolov8n.json')

    # Pre-index ground truth by image_id
    gt_by_img = {}
    for ann in coco_gt_dict['annotations']:
        gt_by_img.setdefault(ann['image_id'], []).append(ann)

    # Pre-index predictions by image_id
    def index_by_img(preds):
        d = {}
        for p in preds:
            d.setdefault(p['image_id'], []).append(p)
        return d

    p_yolo_m = index_by_img(preds_yolov8m)
    p_frcnn = index_by_img(preds_fasterrcnn)
    p_yolo_n = index_by_img(preds_yolov8n)

    categories = coco_gt_dict['categories']

    # Pre-evaluate baseline metrics on original full test set
    def eval_on_coco(coco_gt_obj, pred_list):
        if not pred_list:
            return 0.0, 0.0, 0.0
        coco_dt = coco_gt_obj.loadRes(pred_list)
        ev = COCOeval(coco_gt_obj, coco_dt, 'bbox')
        ev.params.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return float(ev.stats[0]), float(ev.stats[1]), float(ev.stats[2])

    print("\n--- Baseline Metrics (Full Test Set) ---")
    orig_coco = COCO(gt_path)
    base_mAP_ym, base_ap50_ym, base_ap75_ym = eval_on_coco(orig_coco, preds_yolov8m)
    base_mAP_fr, base_ap50_fr, base_ap75_fr = eval_on_coco(orig_coco, preds_fasterrcnn)
    base_mAP_yn, base_ap50_yn, base_ap75_yn = eval_on_coco(orig_coco, preds_yolov8n)

    print(f"YOLOv8m:     mAP={base_mAP_ym:.4f}, AP50={base_ap50_ym:.4f}, AP75={base_ap75_ym:.4f}")
    print(f"Faster RCNN: mAP={base_mAP_fr:.4f}, AP50={base_ap50_fr:.4f}, AP75={base_ap75_fr:.4f}")
    print(f"YOLOv8n:     mAP={base_mAP_yn:.4f}, AP50={base_ap50_yn:.4f}, AP75={base_ap75_yn:.4f}")

    diff_ym_fr_map = []
    diff_ym_fr_ap50 = []
    diff_ym_fr_ap75 = []

    diff_ym_yn_map = []
    diff_ym_yn_ap50 = []
    diff_ym_yn_ap75 = []

    t0 = time.time()
    print(f"\nRunning {num_bootstraps} paired bootstrap resamples...")

    # Suppress pycocotools summarize output during bootstrap
    import io
    from contextlib import redirect_stdout

    for b in range(num_bootstraps):
        if (b + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Iteration {b+1}/{num_bootstraps} ({elapsed:.1f}s)")

        # Sample image IDs with replacement
        sample_ids = np.random.choice(img_ids, size=n_images, replace=True)

        # Build surrogate COCO GT & predictions for this sample
        sample_images = []
        sample_annotations = []
        sample_p_ym = []
        sample_p_fr = []
        sample_p_yn = []

        ann_id_counter = 1
        for surrogate_id, orig_id in enumerate(sample_ids):
            sample_images.append({'id': surrogate_id, 'file_name': f'surr_{surrogate_id}.jpg'})
            
            # Ground truth annotations
            for ann in gt_by_img.get(orig_id, []):
                new_ann = ann.copy()
                new_ann['id'] = ann_id_counter
                new_ann['image_id'] = surrogate_id
                sample_annotations.append(new_ann)
                ann_id_counter += 1

            # Predictions
            for p in p_yolo_m.get(orig_id, []):
                new_p = p.copy()
                new_p['image_id'] = surrogate_id
                sample_p_ym.append(new_p)

            for p in p_frcnn.get(orig_id, []):
                new_p = p.copy()
                new_p['image_id'] = surrogate_id
                sample_p_fr.append(new_p)

            for p in p_yolo_n.get(orig_id, []):
                new_p = p.copy()
                new_p['image_id'] = surrogate_id
                sample_p_yn.append(new_p)

        surrogate_gt = {
            'images': sample_images,
            'annotations': sample_annotations,
            'categories': categories
        }

        with redirect_stdout(io.StringIO()):
            # Create in-memory COCO object
            sub_coco = COCO()
            sub_coco.dataset = surrogate_gt
            sub_coco.createIndex()

            mAP_ym, ap50_ym, ap75_ym = eval_on_coco(sub_coco, sample_p_ym)
            mAP_fr, ap50_fr, ap75_fr = eval_on_coco(sub_coco, sample_p_fr)
            mAP_yn, ap50_yn, ap75_yn = eval_on_coco(sub_coco, sample_p_yn)

        diff_ym_fr_map.append(mAP_ym - mAP_fr)
        diff_ym_fr_ap50.append(ap50_ym - ap50_fr)
        diff_ym_fr_ap75.append(ap75_ym - ap75_fr)

        diff_ym_yn_map.append(mAP_ym - mAP_yn)
        diff_ym_yn_ap50.append(ap50_ym - ap50_yn)
        diff_ym_yn_ap75.append(ap75_ym - ap75_yn)

    def compute_stats(diffs, orig_delta):
        diffs = np.array(diffs)
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        # Empirical two-sided p-value
        p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        return {
            'orig_delta': round(float(orig_delta), 4),
            'boot_mean': round(float(mean_diff), 4),
            'boot_std': round(float(std_diff), 4),
            'ci_lower_95': round(float(ci_lower), 4),
            'ci_upper_95': round(float(ci_upper), 4),
            'p_value': round(float(p_val), 4)
        }

    results = {
        'comparison_yolov8m_vs_fasterrcnn': {
            'mAP_5_95': compute_stats(diff_ym_fr_map, base_mAP_ym - base_mAP_fr),
            'AP_50': compute_stats(diff_ym_fr_ap50, base_ap50_ym - base_ap50_fr),
            'AP_75': compute_stats(diff_ym_fr_ap75, base_ap75_ym - base_ap75_fr)
        },
        'comparison_yolov8m_vs_yolov8n': {
            'mAP_5_95': compute_stats(diff_ym_yn_map, base_mAP_ym - base_mAP_yn),
            'AP_50': compute_stats(diff_ym_yn_ap50, base_ap50_ym - base_ap50_yn),
            'AP_75': compute_stats(diff_ym_yn_ap75, base_ap75_ym - base_ap75_yn)
        }
    }

    # Save JSON and CSV
    with open('results_v2/bootstrap_ci.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open('run2/bootstrap_ci.json', 'w') as f:
        json.dump(results, f, indent=2)

    rows = []
    for comp_name, metrics in results.items():
        pair_str = "YOLOv8m vs. Faster R-CNN" if "fasterrcnn" in comp_name else "YOLOv8m vs. YOLOv8n"
        for metric_name, stats in metrics.items():
            rows.append({
                "Comparison": pair_str,
                "Metric": metric_name,
                "Delta (Point)": stats['orig_delta'],
                "Bootstrap Mean": stats['boot_mean'],
                "Bootstrap Std": stats['boot_std'],
                "95% CI Lower": stats['ci_lower_95'],
                "95% CI Upper": stats['ci_upper_95'],
                "p-value": stats['p_value']
            })

    df = pd.DataFrame(rows)
    df.to_csv('results_v2/bootstrap_ci.csv', index=False)
    df.to_csv('run2/bootstrap_ci.csv', index=False)
    print("\n================ Bootstrap Confidence Interval Results ================")
    print(df.to_string())

    # Generate LaTeX table
    tex = r"""\begin{table}[htbp]
\centering
\caption{Paired Bootstrap Confidence Intervals ($B=1{,}000$ iterations with replacement over $N=145$ test images) comparing YOLOv8m against Faster R-CNN and YOLOv8n. We report the point estimate difference $\Delta$, the empirical 95\% confidence interval $[\text{CI}_{2.5\%}, \text{CI}_{97.5\%}]$, and the two-tailed bootstrap empirical $p$-value testing $H_0: \Delta = 0$.}
\label{tab:bootstrap_ci}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{llcccc}
\toprule
\textbf{Comparison} & \textbf{Metric} & $\bm{\Delta}$ \textbf{(Point)} & \textbf{Bootstrap Mean} & \textbf{95\% Confidence Interval} & $\bm{p}$\textbf{-value} \\
\midrule
"""
    for _, r in df.iterrows():
        p_str = "< 0.001" if r['p-value'] < 0.001 else f"{r['p-value']:.3f}"
        tex += f"{r['Comparison']} & {r['Metric'].replace('_', '\\_')} & {r['Delta (Point)']:+.3f} & {r['Bootstrap Mean']:+.3f} & [{r['95% CI Lower']:+.3f}, {r['95% CI Upper']:+.3f}] & {p_str} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open('run2/table_bootstrap_ci.tex', 'w') as f:
        f.write(tex)
    with open('results_v2/table_bootstrap_ci.tex', 'w') as f:
        f.write(tex)
    print("\nSaved LaTeX bootstrap CI table to run2/table_bootstrap_ci.tex and results_v2/table_bootstrap_ci.tex")

if __name__ == '__main__':
    run_paired_bootstrap(num_bootstraps=1000, seed=42)
