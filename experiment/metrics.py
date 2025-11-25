# -*- coding: utf-8 -*-
# metrics_utils.py
import os, json, glob, time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def list_images(dirp: str) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dirp, e)))
    return sorted(files)

def file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / (1024*1024), 2)

def _percentile(arr, p):
    if len(arr) == 0: return float("nan")
    return float(np.percentile(np.array(arr, dtype=np.float64), p))

def compute_latency(model, img_paths: List[str], imgsz: int, device: str,
                    warmup: int = 10, runs: int = 100) -> Dict[str, float]:
    """
    batch=1 的端到端推理时延统计：mean / P95
    """
    if len(img_paths) == 0:
        return {"latency_mean_ms": float("nan"), "latency_p95_ms": float("nan"), "runs": 0}

    # 预热
    sample = img_paths[0]
    for _ in range(warmup):
        _ = model.predict(source=sample, imgsz=imgsz, device=device, verbose=False, conf=0.001)
        if torch.cuda.is_available() and device != 'cpu':
            torch.cuda.synchronize()

    # 正式计时
    lat = []
    rng = np.random.default_rng(7)
    for i in range(runs):
        src = str(rng.choice(img_paths))
        t0 = time.perf_counter()
        _ = model.predict(source=src, imgsz=imgsz, device=device, verbose=False, conf=0.001)
        if torch.cuda.is_available() and device != 'cpu':
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)  # ms

    return {
        "latency_mean_ms": float(np.mean(lat)),
        "latency_p95_ms": _percentile(lat, 95.0),
        "runs": runs
    }

def extract_ultralytics_metrics(val_res) -> Dict[str, float]:
    """
    兼容不同 ultralytics 版本，提取 COCO 口径的指标。
    优先从 val_res.box 读取（map, map50, mp, mr），否则尝试 results_dict。
    """
    out = {"map5095": np.nan, "map50": np.nan, "precision": np.nan, "recall": np.nan}
    try:
        box = getattr(val_res, "box", None)
        if box is not None:
            out["map5095"] = float(getattr(box, "map", np.nan))
            out["map50"]   = float(getattr(box, "map50", np.nan))
            out["precision"] = float(getattr(box, "mp", np.nan))
            out["recall"]    = float(getattr(box, "mr", np.nan))
            return out
        # fallback
        rd = getattr(val_res, "results_dict", None)
        if isinstance(rd, dict):
            # 常见 key：metrics/mAP50-95(B) 等
            def _get(*keys, default=np.nan):
                for k in keys:
                    if k in rd: return float(rd[k])
                return default
            out["map5095"] = _get("metrics/mAP50-95(B)", "metrics/mAP50-95")
            out["map50"]   = _get("metrics/mAP50(B)", "metrics/mAP50")
            out["precision"] = _get("metrics/precision(B)", "metrics/precision")
            out["recall"]    = _get("metrics/recall(B)", "metrics/recall")
    except Exception:
        pass
    return out

def plot_training_curves(results_csv: str, out_dir: str):
    """从 Ultralytics results.csv 画关键训练曲线"""
    if not os.path.exists(results_csv):
        print(f"[WARN] results.csv not found: {results_csv}")
        return
    ensure_dir(out_dir)
    df = pd.read_csv(results_csv)

    # mAP@[0.5:0.95]
    plt.figure()
    col = 'metrics/mAP50-95(B)' if 'metrics/mAP50-95(B)' in df.columns else 'metrics/mAP50-95'
    if col in df.columns:
        plt.plot(df['epoch'], df[col])
        plt.xlabel('epoch'); plt.ylabel('mAP@[0.5:0.95]')
        plt.title('Validation mAP@[0.5:0.95] over epochs')
        plt.grid(True, linestyle='--', linewidth=0.5); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'curve_map5095.png'), dpi=180)
    plt.close()

    # Precision / Recall
    for metric_col, fname, label in [
        ('metrics/precision(B)', 'curve_precision.png', 'Precision'),
        ('metrics/recall(B)',    'curve_recall.png',    'Recall')
    ]:
        if metric_col in df.columns:
            plt.figure()
            plt.plot(df['epoch'], df[metric_col])
            plt.xlabel('epoch'); plt.ylabel(label)
            plt.title(f'Validation {label} over epochs')
            plt.grid(True, linestyle='--', linewidth=0.5); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=180)
            plt.close()

    # Loss
    for metric_col, fname, label in [
        ('train/box_loss', 'curve_boxloss.png', 'Train Box Loss'),
        ('train/cls_loss', 'curve_clsloss.png', 'Train Cls Loss'),
        ('train/dfl_loss', 'curve_dflloss.png', 'Train DFL Loss')
    ]:
        if metric_col in df.columns:
            plt.figure()
            plt.plot(df['epoch'], df[metric_col])
            plt.xlabel('epoch'); plt.ylabel(label)
            plt.title(label + ' over epochs')
            plt.grid(True, linestyle='--', linewidth=0.5); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=180)
            plt.close()

def save_summary_json(save_path: str, data: Dict[str, Any]):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def latex_row_for(summary: Dict[str, Any], model_label: str) -> str:
    def f2(x):
        try: return f"{float(x):.2f}"
        except: return "-"
    return (
        f"{model_label} & "
        f"{f2(summary['metrics']['mAP@[0.5:0.95]'])} & "
        f"{f2(summary['metrics']['mAP@0.5'])} & "
        f"{f2(summary['latency_ms']['mean'])} & "
        f"{f2(summary['latency_ms']['p95'])} & "
        f"{f2(summary['size_mb'])} \\\\"
    )

def plot_global_bars(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    plt.figure()
    plt.bar(df["model"], df["mAP@[0.5:0.95]"])
    plt.ylabel("mAP@[0.5:0.95]"); plt.title("Model Comparison on mAP@[0.5:0.95]")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cmp_map5095.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(df["model"], df["latency_mean_ms"])
    plt.ylabel("Mean Latency (ms)"); plt.title("Model Comparison on Mean Latency (ms)")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cmp_latency_mean.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(df["model"], df["size_mb"])
    plt.ylabel("Size (MB)"); plt.title("Model Size (MB)")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cmp_size_mb.png"), dpi=200)
    plt.close()
