import os
import json
import random
import yaml
from pathlib import Path
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_common_config(config_path="configs/common.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Deterministic seed set to: {seed}")

def save_resolved_hyperparams(model_name, seed, params, logs_dir="logs_v2"):
    os.makedirs(logs_dir, exist_ok=True)
    out_path = Path(logs_dir) / f"{model_name}_seed{seed}_hparams.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
    print(f"[HParams] Logged resolved hyperparameters to {out_path}")
    return str(out_path)

def setup_environment():
    os.makedirs('checkpoints_v2', exist_ok=True)
    os.makedirs('results_v2/figures', exist_ok=True)
    os.makedirs('logs_v2', exist_ok=True)

    env_path = 'results_v2/env_info.txt'
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available:  {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Model:       {torch.cuda.get_device_name(0)}\n")
            f.write(f"VRAM (MB):       {torch.cuda.get_device_properties(0).total_memory // 1024**2}\n")

    print(f"[Setup] Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
