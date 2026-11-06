#!/usr/bin/env python3
"""
run2/run_interleaved_benchmark.py - Interleaved 3-Repeat Speed Benchmark Across Separate Subprocesses

Runs all five models across 3 interleaved repeats in separate python subprocesses.
Logs GPU telemetry, computes mean and range of median latencies across repeats,
and outputs updated LaTeX tables and summaries.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd

def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout.strip()

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    print("="*60)
    print("STARTING INTERLEAVED 3-REPEAT BENCHMARK ACROSS SEPARATE PROCESSES")
    print("="*60)

    # 1. Log GPU Telemetry
    gpu_info = run_cmd("nvidia-smi --query-gpu=name,driver_version,pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw --format=csv,noheader")
    print(f"Initial GPU Status: {gpu_info}")

    python_exe = sys.executable

    tmp_dir = Path("results_v2/interleaved_raw")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Models and modes to run
    # For YOLO, native is 640x640, so native = uniform.
    jobs_per_repeat = [
        ('yolov8m', 'native'),
        ('yolov8n', 'native'),
        ('fasterrcnn', 'native'),
        ('fasterrcnn', 'uniform640'),
        ('detr', 'native'),
        ('detr', 'uniform640'),
        ('efficientdet', 'native'),
        ('efficientdet', 'uniform640'),
    ]

    all_records = []

    for repeat in range(1, 4):
        print(f"\n>>> Starting Repeat {repeat} of 3 (Interleaved Order) <<<")
        for model_name, mode in jobs_per_repeat:
            out_file = tmp_dir / f"{model_name}_{mode}_r{repeat}.json"
            cmd = f'"{python_exe}" run2/benchmark_worker.py --model {model_name} --mode {mode} --repeat {repeat} --output "{out_file}"'
            print(f"Executing: {model_name} ({mode}) Repeat {repeat} in fresh process...")
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                print(f"ERROR: Job failed for {model_name} {mode} R{repeat}!")
                sys.exit(1)

            with open(out_file, 'r') as f:
                data = json.load(f)
                all_records.append(data)
            
            # Brief pause to stabilize GPU thermals
            time.sleep(1.0)

    # For YOLO, uniform640 is identical to native
    for r in range(1, 4):
        for ym in ['yolov8m', 'yolov8n']:
            nat = [x for x in all_records if x['model'] == ym and x['mode'] == 'native' and x['repeat'] == r][0]
            uni = dict(nat)
            uni['mode'] = 'uniform640'
            all_records.append(uni)

    # Save all raw records
    with open('results_v2/interleaved_all_runs.json', 'w') as f:
        json.dump(all_records, f, indent=2)

    # Compute aggregation: Mean and Range of Medians across repeats
    df = pd.DataFrame(all_records)

    summary_rows = []
    models_order = ['yolov8m', 'yolov8n', 'fasterrcnn', 'detr', 'efficientdet']
    modes_order = ['native', 'uniform640']

    for m in models_order:
        for mode in modes_order:
            sub = df[(df['model'] == m) & (df['mode'] == mode)]
            mod_meds = sub['mod_median'].values
            e2e_meds = sub['e2e_median'].values
            mod_iqrs = sub['mod_iqr'].values
            e2e_iqrs = sub['e2e_iqr'].values

            mean_mod_med = float(np.mean(mod_meds))
            min_mod_med = float(np.min(mod_meds))
            max_mod_med = float(np.max(mod_meds))
            range_mod_med = max_mod_med - min_mod_med

            mean_e2e_med = float(np.mean(e2e_meds))
            min_e2e_med = float(np.min(e2e_meds))
            max_e2e_med = float(np.max(e2e_meds))
            range_e2e_med = max_e2e_med - min_e2e_med

            mean_mod_iqr = float(np.mean(mod_iqrs))
            mean_e2e_iqr = float(np.mean(e2e_iqrs))

            mod_fps = 1000.0 / mean_mod_med if mean_mod_med > 0 else 0
            e2e_fps = 1000.0 / mean_e2e_med if mean_e2e_med > 0 else 0

            summary_rows.append({
                'model': m,
                'mode': mode,
                'mod_median_mean': round(mean_mod_med, 2),
                'mod_median_min': round(min_mod_med, 2),
                'mod_median_max': round(max_mod_med, 2),
                'mod_median_range': round(range_mod_med, 2),
                'mod_iqr_mean': round(mean_mod_iqr, 2),
                'mod_fps': round(mod_fps, 1),
                'e2e_median_mean': round(mean_e2e_med, 2),
                'e2e_median_min': round(min_e2e_med, 2),
                'e2e_median_max': round(max_e2e_med, 2),
                'e2e_median_range': round(range_e2e_med, 2),
                'e2e_iqr_mean': round(mean_e2e_iqr, 2),
                'e2e_fps': round(e2e_fps, 1),
                'r1_mod_med': round(mod_meds[0], 2),
                'r2_mod_med': round(mod_meds[1], 2),
                'r3_mod_med': round(mod_meds[2], 2),
                'r1_e2e_med': round(e2e_meds[0], 2),
                'r2_e2e_med': round(e2e_meds[1], 2),
                'r3_e2e_med': round(e2e_meds[2], 2),
            })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv('results_v2/interleaved_summary.csv', index=False)
    with open('results_v2/interleaved_summary.json', 'w') as f:
        json.dump(summary_rows, f, indent=2)

    print("\n" + "="*80)
    print("INTERLEAVED 3-REPEAT SPEED BENCHMARK SUMMARY (N=145 images, batch size 1, FP32)")
    print("="*80)
    for r in summary_rows:
        print(f"Model: {r['model']:12s} | Mode: {r['mode']:10s} | Forward Med: {r['mod_median_mean']:5.2f} ms (range [{r['mod_median_min']:.2f}, {r['mod_median_max']:.2f}], delta={r['mod_median_range']:.2f}) -> {r['mod_fps']:5.1f} FPS | E2E Med: {r['e2e_median_mean']:5.2f} ms (range [{r['e2e_median_min']:.2f}, {r['e2e_median_max']:.2f}], delta={r['e2e_median_range']:.2f}) -> {r['e2e_fps']:5.1f} FPS")

    final_gpu = run_cmd("nvidia-smi --query-gpu=name,driver_version,pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw --format=csv,noheader")
    print(f"\nFinal GPU Status: {final_gpu}")

if __name__ == '__main__':
    main()
