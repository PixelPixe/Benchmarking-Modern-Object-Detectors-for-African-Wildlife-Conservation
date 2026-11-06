#!/usr/bin/env python3
"""
dataset_stats.py - Comprehensive Dataset Statistics Generator for WildCareAI2

Generates:
1. Table of images and per-class instances across Train / Val / Test splits (CSV & LaTeX)
2. Table of Ground Truth instances in COCO scale bins (Small, Medium, Large) for the Test set,
   both overall and per class (CSV & LaTeX).
3. Scale distributions across all splits for verification.

All metrics are computed directly from the authoritative COCO ground-truth annotations:
- coco_annotations/instances_train.json
- coco_annotations/instances_val.json
- coco_annotations/instances_test.json
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']
SPLITS = ['train', 'val', 'test']

# Standard COCO area thresholds (pixels squared)
# Small: area < 32^2 = 1024 px^2
# Medium: 32^2 <= area < 96^2 = [1024, 9216) px^2
# Large: area >= 96^2 = 9216 px^2
SMALL_THRESH = 32 ** 2   # 1024
LARGE_THRESH = 96 ** 2   # 9216

def get_scale_bin(area):
    if area < SMALL_THRESH:
        return 'Small (<32^2 px^2)'
    elif area < LARGE_THRESH:
        return 'Medium (32^2-96^2 px^2)'
    else:
        return 'Large (>=96^2 px^2)'

def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    
    out_dirs = [Path('run2'), Path('results_v2')]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Dataset Split and Class Distribution Statistics
    # ---------------------------------------------------------
    split_stats = []
    
    for split in SPLITS:
        ann_path = Path(f'coco_annotations/instances_{split}.json')
        with open(ann_path, 'r') as f:
            data = json.load(f)
            
        n_images = len(data['images'])
        cls_counts = defaultdict(int)
        
        for ann in data['annotations']:
            cls_counts[ann['category_id']] += 1
            
        total_instances = len(data['annotations'])
        
        row = {
            'Split': split.capitalize(),
            'Images': n_images,
            'Buffalo': cls_counts[0],
            'Elephant': cls_counts[1],
            'Rhino': cls_counts[2],
            'Zebra': cls_counts[3],
            'Total Instances': total_instances
        }
        split_stats.append(row)

    # Add Total Row
    df_splits = pd.DataFrame(split_stats)
    totals = {
        'Split': 'Total',
        'Images': df_splits['Images'].sum(),
        'Buffalo': df_splits['Buffalo'].sum(),
        'Elephant': df_splits['Elephant'].sum(),
        'Rhino': df_splits['Rhino'].sum(),
        'Zebra': df_splits['Zebra'].sum(),
        'Total Instances': df_splits['Total Instances'].sum()
    }
    df_splits = pd.concat([df_splits, pd.DataFrame([totals])], ignore_index=True)

    # Save CSVs
    for d in out_dirs:
        df_splits.to_csv(d / 'dataset_split_stats.csv', index=False)

    # LaTeX Table for Dataset Splits
    latex_splits = r"""\begin{table}[htbp]
\centering
\caption{Dataset Partition and Per-Class Ground-Truth Instance Counts}
\label{tab:dataset_splits}
\begin{tabular}{lcccccc}
\toprule
\textbf{Split} & \textbf{Images} & \textbf{Buffalo} & \textbf{Elephant} & \textbf{Rhino} & \textbf{Zebra} & \textbf{Total Instances} \\
\midrule
"""
    for _, r in df_splits.iterrows():
        is_total = r['Split'] == 'Total'
        if is_total:
            latex_splits += r"\midrule" + "\n"
            latex_splits += f"\\textbf{{{r['Split']}}} & \\textbf{{{r['Images']:,}}} & \\textbf{{{r['Buffalo']:,}}} & \\textbf{{{r['Elephant']:,}}} & \\textbf{{{r['Rhino']:,}}} & \\textbf{{{r['Zebra']:,}}} & \\textbf{{{r['Total Instances']:,}}} \\\\\n"
        else:
            latex_splits += f"{r['Split']} & {r['Images']:,} & {r['Buffalo']:,} & {r['Elephant']:,} & {r['Rhino']:,} & {r['Zebra']:,} & {r['Total Instances']:,} \\\\\n"
    latex_splits += r"""\bottomrule
\end{tabular}
\end{table}
"""
    for d in out_dirs:
        with open(d / 'dataset_split_stats.tex', 'w', encoding='utf-8') as f:
            f.write(latex_splits)

    # ---------------------------------------------------------
    # 2. Test Set Scale Distribution (COCO Bins)
    # ---------------------------------------------------------
    test_path = Path('coco_annotations/instances_test.json')
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    scale_counts = {
        'Small (<32^2 px^2)': defaultdict(int),
        'Medium (32^2-96^2 px^2)': defaultdict(int),
        'Large (>=96^2 px^2)': defaultdict(int)
    }
    scale_totals = defaultdict(int)

    for ann in test_data['annotations']:
        area = ann['area']
        cat = ann['category_id']
        bin_name = get_scale_bin(area)
        scale_counts[bin_name][cat] += 1
        scale_totals[bin_name] += 1

    scale_rows = []
    order = ['Small (<32^2 px^2)', 'Medium (32^2-96^2 px^2)', 'Large (>=96^2 px^2)']
    for b in order:
        tot = scale_totals[b]
        pct = (tot / len(test_data['annotations'])) * 100
        scale_rows.append({
            'Scale Category': b,
            'Area Definition': '< 1,024 px^2' if 'Small' in b else ('1,024 - 9,216 px^2' if 'Medium' in b else '>= 9,216 px^2'),
            'Buffalo': scale_counts[b][0],
            'Elephant': scale_counts[b][1],
            'Rhino': scale_counts[b][2],
            'Zebra': scale_counts[b][3],
            'Total Instances': tot,
            'Share (%)': f"{pct:.2f}%"
        })

    df_scale = pd.DataFrame(scale_rows)
    # Add Total Row
    scale_tot_row = {
        'Scale Category': 'Total',
        'Area Definition': 'All Scales',
        'Buffalo': df_scale['Buffalo'].sum(),
        'Elephant': df_scale['Elephant'].sum(),
        'Rhino': df_scale['Rhino'].sum(),
        'Zebra': df_scale['Zebra'].sum(),
        'Total Instances': len(test_data['annotations']),
        'Share (%)': "100.00%"
    }
    df_scale = pd.concat([df_scale, pd.DataFrame([scale_tot_row])], ignore_index=True)

    for d in out_dirs:
        df_scale.to_csv(d / 'test_scale_stats.csv', index=False)

    # LaTeX Table for Scale Bins
    latex_scale = r"""\begin{table}[htbp]
\centering
\caption{Ground-Truth Instance Distribution by COCO Area Scale (Test Set, $N=145$ images)}
\label{tab:test_scale_distribution}
\begin{tabular}{lccccccr}
\toprule
\textbf{Scale Category} & \textbf{Area Range ($px^2$)} & \textbf{Buffalo} & \textbf{Elephant} & \textbf{Rhino} & \textbf{Zebra} & \textbf{Total} & \textbf{Share (\%)} \\
\midrule
"""
    latex_name_map = {
        'Small (<32^2 px^2)': r'Small ($<32^2\,\text{px}^2$)',
        'Medium (32^2-96^2 px^2)': r'Medium ($32^2\text{--}96^2\,\text{px}^2$)',
        'Large (>=96^2 px^2)': r'Large ($\ge 96^2\,\text{px}^2$)',
        'Total': r'\textbf{Total}'
    }
    latex_area_map = {
        '< 1,024 px^2': r'$< 1{,}024\,\text{px}^2$',
        '1,024 - 9,216 px^2': r'$1{,}024\text{--}9{,}216\,\text{px}^2$',
        '>= 9,216 px^2': r'$\ge 9{,}216\,\text{px}^2$',
        'All Scales': 'All Scales'
    }

    for _, r in df_scale.iterrows():
        is_total = r['Scale Category'] == 'Total'
        cat_name = latex_name_map.get(r['Scale Category'], r['Scale Category'])
        area_def = latex_area_map.get(r['Area Definition'], r['Area Definition'])
        share_str = str(r['Share (%)']).replace('%', r'\%')
        if is_total:
            latex_scale += r"\midrule" + "\n"
            latex_scale += f"{cat_name} & \\textbf{{{area_def}}} & \\textbf{{{r['Buffalo']}}} & \\textbf{{{r['Elephant']}}} & \\textbf{{{r['Rhino']}}} & \\textbf{{{r['Zebra']}}} & \\textbf{{{r['Total Instances']}}} & \\textbf{{{share_str}}} \\\\\n"
        else:
            latex_scale += f"{cat_name} & {area_def} & {r['Buffalo']} & {r['Elephant']} & {r['Rhino']} & {r['Zebra']} & {r['Total Instances']} & {share_str} \\\\\n"
    latex_scale += r"""\bottomrule
\end{tabular}
\end{table}
"""
    for d in out_dirs:
        with open(d / 'test_scale_stats.tex', 'w', encoding='utf-8') as f:
            f.write(latex_scale)

    print("==============================================================")
    print("[SUCCESS] Dataset statistics successfully written to run2/ and results_v2/")
    print("==============================================================")
    print("\n--- Split Summary ---")
    print(df_splits.to_string(index=False))
    print("\n--- Test Set Scale Summary ---")
    print(df_scale.to_string(index=False))

if __name__ == '__main__':
    main()
