import os
import json
import time
import io
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_high_iou_bootstrap(num_bootstraps=1000, seed=42):
    np.random.seed(seed)
    gt_path = 'coco_annotations/instances_test.json'
    with open(gt_path, 'r') as f:
        coco_gt_dict = json.load(f)

    img_ids = [img['id'] for img in coco_gt_dict['images']]
    n_images = len(img_ids)

    with open('results/predictions_yolov8m.json') as f:
        preds_ym = json.load(f)
    with open('results/predictions_fasterrcnn.json') as f:
        preds_fr = json.load(f)

    gt_by_img = {}
    for ann in coco_gt_dict['annotations']:
        gt_by_img.setdefault(ann['image_id'], []).append(ann)

    def index_by_img(preds):
        d = {}
        for p in preds:
            d.setdefault(p['image_id'], []).append(p)
        return d

    p_ym_by_img = index_by_img(preds_ym)
    p_fr_by_img = index_by_img(preds_fr)
    categories = coco_gt_dict['categories']

    def eval_ap_thresholds(coco_gt_obj, pred_list):
        if not pred_list:
            return 0.0, 0.0
        coco_dt = coco_gt_obj.loadRes(pred_list)
        ev = COCOeval(coco_gt_obj, coco_dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        p = ev.eval['precision']
        s85 = p[7, :, :, 0, 2]
        ap85 = float(np.mean(s85[s85 > -1])) if np.any(s85 > -1) else 0.0
        s90 = p[8, :, :, 0, 2]
        ap90 = float(np.mean(s90[s90 > -1])) if np.any(s90 > -1) else 0.0
        return ap85, ap90

    diff_ap85 = []
    diff_ap90 = []

    print(f"Running {num_bootstraps} bootstrap resamples for AP85 and AP90...")
    t0 = time.time()
    for b in range(num_bootstraps):
        if (b + 1) % 100 == 0:
            print(f"  Iteration {b+1}/{num_bootstraps} ({time.time() - t0:.1f}s)")
        sample_ids = np.random.choice(img_ids, size=n_images, replace=True)

        sample_images = []
        sample_annotations = []
        sample_p_ym = []
        sample_p_fr = []

        ann_id = 1
        for surr_id, orig_id in enumerate(sample_ids):
            sample_images.append({'id': surr_id, 'file_name': f'surr_{surr_id}.jpg'})
            for ann in gt_by_img.get(orig_id, []):
                new_ann = ann.copy()
                new_ann['id'] = ann_id
                new_ann['image_id'] = surr_id
                sample_annotations.append(new_ann)
                ann_id += 1

            for p in p_ym_by_img.get(orig_id, []):
                new_p = p.copy()
                new_p['image_id'] = surr_id
                sample_p_ym.append(new_p)

            for p in p_fr_by_img.get(orig_id, []):
                new_p = p.copy()
                new_p['image_id'] = surr_id
                sample_p_fr.append(new_p)

        surr_gt = {
            'images': sample_images,
            'annotations': sample_annotations,
            'categories': categories
        }

        with redirect_stdout(io.StringIO()):
            sub_coco = COCO()
            sub_coco.dataset = surr_gt
            sub_coco.createIndex()
            ap85_ym, ap90_ym = eval_ap_thresholds(sub_coco, sample_p_ym)
            ap85_fr, ap90_fr = eval_ap_thresholds(sub_coco, sample_p_fr)

        diff_ap85.append(ap85_ym - ap85_fr)
        diff_ap90.append(ap90_ym - ap90_fr)

    def stats(diffs, point_delta):
        arr = np.array(diffs)
        se = float(np.std(arr))
        ci_l = float(np.percentile(arr, 2.5))
        ci_u = float(np.percentile(arr, 97.5))
        # Wald z and p-value
        z = abs(point_delta) / se if se > 0 else 0
        from scipy.stats import norm
        p_val = 2 * (1 - norm.cdf(z))
        return {
            'point_delta': point_delta,
            'se': se,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
            'p_value': p_val
        }

    res85 = stats(diff_ap85, 0.7413 - 0.6113)
    res90 = stats(diff_ap90, 0.5960 - 0.3051)

    print("\n--- RESULTS ---")
    print("AP85:", res85)
    print("AP90:", res90)

    # Save to json/csv
    out = {
        'AP85': res85,
        'AP90': res90
    }
    with open('run2/bootstrap_high_iou.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    run_high_iou_bootstrap()
