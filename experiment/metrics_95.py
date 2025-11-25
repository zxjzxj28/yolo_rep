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
    优先从 val_res.box 读取（map, map50, map75, map95, mp, mr），否则尝试 results_dict。
    
    返回字段：
    - map5095: mAP@[0.5:0.95] (平均)
    - map50: mAP@0.5
    - map75: mAP@0.75
    - map95: mAP@0.95  ★ 新增
    - precision: 精确率
    - recall: 召回率
    """
    out = {
        "map5095": np.nan, 
        "map50": np.nan, 
        "map75": np.nan,      # ★ 新增
        "map95": np.nan,      # ★ 新增
        "precision": np.nan, 
        "recall": np.nan
    }
    
    try:
        box = getattr(val_res, "box", None)
        if box is not None:
            # 标准字段
            out["map5095"] = float(getattr(box, "map", np.nan))
            out["map50"] = float(getattr(box, "map50", np.nan))
            out["map75"] = float(getattr(box, "map75", np.nan))  # ★ 新增
            out["precision"] = float(getattr(box, "mp", np.nan))
            out["recall"] = float(getattr(box, "mr", np.nan))
            
            # ★ 提取 mAP95
            # Ultralytics 的 box.maps 是一个数组，包含 [0.5:0.95:0.05] 的所有 IoU 阈值
            # maps[0] = mAP@0.5, maps[9] = mAP@0.95
            if hasattr(box, "maps") and box.maps is not None:
                try:
                    maps_array = np.array(box.maps)
                    if len(maps_array) >= 10:  # 确保有足够的元素
                        out["map95"] = float(maps_array[9])  # 索引9对应0.95
                except Exception as e:
                    pass  # 如果失败，保持 nan
            
            # 如果 maps 不可用，尝试从 all_ap 中提取
            if np.isnan(out["map95"]) and hasattr(box, "all_ap"):
                try:
                    # all_ap: shape (num_classes, num_iou_thresholds)
                    all_ap = np.array(box.all_ap)
                    if all_ap.ndim == 2 and all_ap.shape[1] >= 10:
                        # 取所有类别在 IoU=0.95 处的平均
                        out["map95"] = float(np.mean(all_ap[:, 9]))
                except Exception:
                    pass
                    
            return out
            
        # fallback：从 results_dict 提取
        rd = getattr(val_res, "results_dict", None)
        if isinstance(rd, dict):
            def _get(*keys, default=np.nan):
                for k in keys:
                    if k in rd: 
                        return float(rd[k])
                return default
            
            out["map5095"] = _get("metrics/mAP50-95(B)", "metrics/mAP50-95")
            out["map50"] = _get("metrics/mAP50(B)", "metrics/mAP50")
            out["map75"] = _get("metrics/mAP75(B)", "metrics/mAP75")  # ★ 新增
            out["precision"] = _get("metrics/precision(B)", "metrics/precision")
            out["recall"] = _get("metrics/recall(B)", "metrics/recall")
            
            # ★ 尝试从 results_dict 提取 mAP95
            out["map95"] = _get("metrics/mAP95(B)", "metrics/mAP95")
            
    except Exception as e:
        print(f"[WARN] 提取指标时出错：{e}")
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

    # ★ 新增：mAP@0.50 和 mAP@0.75 对比图
    plt.figure()
    map50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df.columns else 'metrics/mAP50'
    map75_col = 'metrics/mAP75(B)' if 'metrics/mAP75(B)' in df.columns else 'metrics/mAP75'
    
    if map50_col in df.columns:
        plt.plot(df['epoch'], df[map50_col], label='mAP@0.5', linewidth=2)
    if map75_col in df.columns:
        plt.plot(df['epoch'], df[map75_col], label='mAP@0.75', linewidth=2)
    
    plt.xlabel('epoch'); plt.ylabel('mAP')
    plt.title('Validation mAP at Different IoU Thresholds')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curve_map_comparison.png'), dpi=180)
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
    """
    生成 LaTeX 表格行
    ★ 新增 mAP@0.95 列
    """
    def f2(x):
        try: return f"{float(x):.2f}"
        except: return "-"
    
    def f4(x):
        try: return f"{float(x):.4f}"
        except: return "-"
    
    # 提取指标（支持旧格式）
    metrics = summary.get('metrics', {})
    map5095 = metrics.get('mAP@[0.5:0.95]', np.nan)
    map50 = metrics.get('mAP@0.5', np.nan)
    map75 = metrics.get('mAP@0.75', np.nan)
    map95 = metrics.get('mAP@0.95', np.nan)  # ★ 新增
    
    latency_ms = summary.get('latency_ms', {})
    lat_mean = latency_ms.get('mean', np.nan)
    lat_p95 = latency_ms.get('p95', np.nan)
    
    size_mb = summary.get('size_mb', np.nan)
    
    return (
        f"{model_label} & "
        f"{f2(map50)} & "
        f"{f2(map75)} & "
        f"{f4(map95)} & "     # ★ 新增（用4位小数，因为通常很低）
        f"{f2(map5095)} & "
        f"{f2(lat_mean)} & "
        f"{f2(lat_p95)} & "
        f"{f2(size_mb)} \\\\"
    )

def plot_global_bars(df: pd.DataFrame, out_dir: str):
    """
    ★ 新增 mAP@0.75 和 mAP@0.95 的对比图
    """
    ensure_dir(out_dir)

    # 原有的 mAP@[0.5:0.95] 对比
    plt.figure()
    plt.bar(df["model"], df["mAP@[0.5:0.95]"])
    plt.ylabel("mAP@[0.5:0.95]"); plt.title("Model Comparison on mAP@[0.5:0.95]")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cmp_map5095.png"), dpi=200)
    plt.close()

    # ★ 新增：多个 IoU 阈值的对比（分组柱状图）
    if "mAP@0.5" in df.columns and "mAP@0.75" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df["model"]))
        width = 0.2
        
        bars = []
        labels = []
        
        if "mAP@0.5" in df.columns:
            bars.append(ax.bar(x - width*1.5, df["mAP@0.5"], width, label='mAP@0.5'))
            labels.append('mAP@0.5')
        
        if "mAP@0.75" in df.columns:
            bars.append(ax.bar(x - width*0.5, df["mAP@0.75"], width, label='mAP@0.75'))
            labels.append('mAP@0.75')
        
        if "mAP@0.95" in df.columns:
            bars.append(ax.bar(x + width*0.5, df["mAP@0.95"], width, label='mAP@0.95'))
            labels.append('mAP@0.95')
        
        if "mAP@[0.5:0.95]" in df.columns:
            bars.append(ax.bar(x + width*1.5, df["mAP@[0.5:0.95]"], width, label='mAP@[0.5:0.95]'))
            labels.append('mAP@[0.5:0.95]')
        
        ax.set_ylabel('mAP')
        ax.set_title('Model Comparison Across IoU Thresholds')
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=30, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cmp_map_all_iou.png"), dpi=200)
        plt.close()

    # 原有的延迟对比
    plt.figure()
    plt.bar(df["model"], df["latency_mean_ms"])
    plt.ylabel("Mean Latency (ms)"); plt.title("Model Comparison on Mean Latency (ms)")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cmp_latency_mean.png"), dpi=200)
    plt.close()

    # 原有的模型大小对比
    plt.figure()
    plt.bar(df["model"], df["size_mb"])
    plt.ylabel("Size (MB)"); plt.title("Model Size (MB)")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cmp_size_mb.png"), dpi=200)
    plt.close()