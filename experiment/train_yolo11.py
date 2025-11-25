# -*- coding: utf-8 -*-
# train_yolo11.py  (multi-GPU ready, latency on TEST; data via YAML; full logging to files)
import os, sys, json, random, shutil, contextlib, logging, signal, atexit, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from dataset import DATASET_CFG, prepare_yolo_dataset
from metrics import (
    ensure_dir, list_images, file_size_mb, compute_latency,
    extract_ultralytics_metrics, plot_training_curves,
    save_summary_json, latex_row_for, plot_global_bars
)

# ============== 实验配置（只改这一块） ==============
CFG = {
    "dataset": DATASET_CFG,

    "train": {
        "imgsz": 640,
        "epochs": 180,
        "batch": 48,               # 自动探测最大 batch（多卡会均分）
        "workers": 12,
        "device": "0,1,2",         # 多卡写 "0,1" 之类
        "project": "runs/detect",
        "seed": 42,
        "cos_lr": True,
        "amp": True,
        # 轻量稳健增强（按需增减）
        "multi_scale": True,
        "degrees": 0.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "patience": 5,
        "optimizer": "auto",
        "weight_decay": 5e-4,
        "cache": "ram",
        # ★ 新增：周期性保存配置
        "save_interval": 30,        # 每N个epoch保存一次完整检查点
    },

    "latency": {
        "pool_samples": 200,   # 从 TEST 中抽样的图片数
        "warmup": 10,
        "runs": 200
    },

    "experiments": [
        {"model": "yolo11s.pt", "name": "exp_y11s"},
        {"model": "yolo8n.pt", "name": "exp_y8n"},
    ],

    "predict_vis": {
        "enable": True,
        "num_images": 16,
        "conf": 0.25
    }
}
# ====================================================

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_root_logger = logging.getLogger("train")
_root_logger.setLevel(logging.INFO)

# ======== 全局"应急上下文" =========
_CURRENT = {
    "exp_name": None,          # 当前实验名
    "save_dir": None,          # 当前实验目录 Path
    "model": None,             # 当前 YOLO 模型对象
    "trainer": None,           # Ultralytics 内部 trainer（在 on_train_start 回调里拿）
    "emergency_done": False,   # 避免重复保存
    # ★ 新增：用于跟踪最佳模型
    "best_fitness": -1.0,      # 最佳 fitness（mAP50-95 为主）
    "epoch_metrics": [],       # 每个epoch的指标历史
    "epoch_log_file": None,    # 当前实验的 epoch 日志文件句柄
}

def _add_file_handler_once(logger: logging.Logger, filepath: Path):
    """为当前实验增加一个 FileHandler（只写文件）。"""
    fh = logging.FileHandler(filepath, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    return fh

def _remove_handler(logger: logging.Logger, handler: logging.Handler):
    try:
        logger.removeHandler(handler)
        handler.close()
    except Exception:
        pass

def _ensure_yaml_from_data_dict(data_dict: Dict[str, Any]) -> str:
    """某些 Ultralytics 版本不接受 dict，需要 YAML 路径字符串。"""
    out_root = Path(data_dict["path"])
    y = out_root / "mobilechart.yaml"
    if not y.exists():
        payload = {
            "path": str(out_root),
            "train": data_dict.get("train", "images/train"),
            "val":   data_dict.get("val",   "images/val"),
            "test":  data_dict.get("test",  "images/test"),
            "names": data_dict.get("names", {})
        }
        try:
            import yaml  # type: ignore
            with open(y, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        except Exception:
            txt = (
                f'path: {payload["path"]}\n'
                f'train: {payload["train"]}\n'
                f'val: {payload["val"]}\n'
                f'test: {payload["test"]}\n'
                f'names:\n' +
                "".join([f'  {int(k)}: {v}\n' for k, v in payload["names"].items()])
            )
            y.parent.mkdir(parents=True, exist_ok=True)
            with open(y, "w", encoding="utf-8") as f:
                f.write(txt)
    return str(y)

# ============== 日志 Tee 到文件（只保留文件，不在控制台刷） ==============
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data); s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

@contextlib.contextmanager
def tee_stdout_stderr(log_path: Path):
    """把 stdout/stderr 重定向到当前实验目录的日志文件（追加）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    t = Tee(f)  # 不再输出到控制台
    sys.stdout, sys.stderr = t, t
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.close()

# ============== 完整检查点保存（包含指标） ==============
def _save_checkpoint_with_metrics(
    trainer,
    epoch: int,
    save_dir: Path,
    filename: str = "checkpoint.pth",
    reason: str = "periodic"
):
    """
    保存完整的训练状态，包括：
    - 模型权重
    - 优化器状态
    - epoch 信息
    - 训练指标历史
    - 当前最佳 fitness
    """
    try:
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = weights_dir / filename

        state = {
            "epoch": epoch,
            "epochs": int(trainer.epochs) if hasattr(trainer, "epochs") else None,
            "best_fitness": _CURRENT.get("best_fitness", -1.0),
            "epoch_metrics": _CURRENT.get("epoch_metrics", []),
            "reason": reason,
        }

        # 保存模型权重
        if hasattr(trainer, "model") and trainer.model is not None:
            state["model_state_dict"] = trainer.model.state_dict()
        
        # 保存优化器状态
        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            state["optimizer_state_dict"] = trainer.optimizer.state_dict()
        
        # 保存学习率调度器状态
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            try:
                state["scheduler_state_dict"] = trainer.scheduler.state_dict()
            except Exception:
                pass

        # 保存 EMA 状态（如果有）
        if hasattr(trainer, "ema") and trainer.ema is not None:
            try:
                state["ema_state_dict"] = trainer.ema.ema.state_dict()
            except Exception:
                pass

        torch.save(state, ckpt_path)
        
        # 同时保存一份纯权重文件（方便直接加载推理）
        if "model_state_dict" in state:
            weights_path = weights_dir / filename.replace(".pth", ".pt")
            torch.save(state["model_state_dict"], weights_path)

        _root_logger.info(f"[CHECKPOINT] 已保存：{ckpt_path} (reason={reason}, epoch={epoch})")
        
        # 保存指标历史到 JSON
        metrics_path = save_dir / f"metrics_history_epoch{epoch:03d}.json"
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epoch": epoch,
                    "best_fitness": state["best_fitness"],
                    "metrics_history": state["epoch_metrics"]
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _root_logger.warning(f"保存指标历史失败：{e}")

        return True
    except Exception as e:
        _root_logger.error(f"[CHECKPOINT] 保存失败：{e}\n{traceback.format_exc()}")
        return False

# ============== 训练过程的"精简行"日志 + 周期性保存 ==============
def _compact_epoch_line(trainer):
    """每个 epoch 结束时的回调：记录指标、保存日志、周期性检查点"""
    m = getattr(trainer, "metrics", {}) or {}
    def g_any(keys, default=0.0):
        for k in keys:
            v = m.get(k, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return default

    try:
        box_l, cls_l, dfl_l = [float(x) for x in trainer.loss_items[:3]]
    except Exception:
        box_l = cls_l = dfl_l = float("nan")

    try:
        mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
    except Exception:
        mem_gb = float("nan")

    try:
        lr = trainer.optimizer.param_groups[0]['lr']
    except Exception:
        lr = float("nan")

    e = int(trainer.epoch) + 1
    E = int(trainer.epochs)
    bs = int(getattr(trainer, "batch_size", 0) or trainer.args.batch)
    imgsz = int(trainer.args.imgsz)

    # 提取关键指标（兼容多版本键名）
    precision = g_any(['metrics/precision(B)', 'metrics/precision'])
    recall    = g_any(['metrics/recall(B)',    'metrics/recall'])
    map50     = g_any(['metrics/mAP50(B)',     'metrics/mAP50'])
    map5095   = g_any(['metrics/mAP50-95(B)',  'metrics/mAP50-95', 'metrics/mAP50-95(B/mAP)'])

    # 记录到历史
    epoch_record = {
        "epoch": e,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
        "loss_box": box_l,
        "loss_cls": cls_l,
        "loss_dfl": dfl_l,
        "lr": lr,
        "mem_gb": mem_gb
    }
    _CURRENT["epoch_metrics"].append(epoch_record)

    # 构建日志行
    line = (f"E{e:03d}/{E:03d} | bs={bs:<3d} | sz={imgsz:<4d} "
            f"| P={precision:.3f} R={recall:.3f} "
            f"| mAP50={map50:.3f} mAP50-95={map5095:.3f} "
            f"| loss: box={box_l:.3f} cls={cls_l:.3f} dfl={dfl_l:.3f} "
            f"| lr={lr:.5f} | mem~{mem_gb:.1f}GB")

    # 写入精简日志文件
    try:
        save_dir = Path(trainer.save_dir)
        with open(save_dir / "epoch_compact.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

    # ★ 写入详细的 epoch 日志文件（每个 epoch 一行 JSON）
    try:
        save_dir = Path(trainer.save_dir)
        with open(save_dir / "epoch_detailed.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
    except Exception as e:
        _root_logger.warning(f"写入详细 epoch 日志失败：{e}")

    _root_logger.info(line)

    # ★ 周期性保存检查点（每 N 个 epoch）
    save_interval = CFG["train"].get("save_interval", 30)
    if e % save_interval == 0:
        try:
            save_dir = Path(trainer.save_dir)
            _save_checkpoint_with_metrics(
                trainer, e, save_dir,
                filename=f"checkpoint_epoch{e:03d}.pth",
                reason=f"periodic_every_{save_interval}"
            )
        except Exception as ex:
            _root_logger.error(f"[CHECKPOINT] 周期性保存失败 (epoch={e})：{ex}")

    # ★ 跟踪并保存最佳模型
    # fitness 定义：以 mAP50-95 为主，也可以自定义组合
    current_fitness = map5095  # 可以调整为 0.1*map50 + 0.9*map5095 等
    
    if current_fitness > _CURRENT["best_fitness"]:
        _CURRENT["best_fitness"] = current_fitness
        try:
            save_dir = Path(trainer.save_dir)
            # 保存最佳模型（完整检查点）
            _save_checkpoint_with_metrics(
                trainer, e, save_dir,
                filename="best_fitness.pth",
                reason=f"best_fitness={current_fitness:.4f}"
            )
            _root_logger.info(f"[BEST] 新最佳模型！fitness={current_fitness:.4f} @ epoch={e}")
        except Exception as ex:
            _root_logger.error(f"[BEST] 保存最佳模型失败：{ex}")

def _on_train_start(trainer):
    """训练开始时的回调"""
    # 捕获内部 trainer 与 save_dir，供应急保存使用
    _CURRENT["trainer"] = trainer
    try:
        _CURRENT["save_dir"] = Path(trainer.save_dir)
    except Exception:
        pass
    
    # 重置最佳 fitness 和指标历史
    _CURRENT["best_fitness"] = -1.0
    _CURRENT["epoch_metrics"] = []
    
    _root_logger.info(f"[TRAIN] 训练开始，目标 epochs={trainer.epochs}")

def _on_train_end(trainer):
    """训练正常结束时的回调"""
    try:
        save_dir = Path(trainer.save_dir)
        final_epoch = int(trainer.epoch) + 1
        
        # 保存最终状态
        _save_checkpoint_with_metrics(
            trainer, final_epoch, save_dir,
            filename="final_checkpoint.pth",
            reason="training_completed"
        )
        
        # 保存完整的指标历史
        metrics_summary_path = save_dir / "training_metrics_complete.json"
        with open(metrics_summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_epochs": final_epoch,
                "best_fitness": _CURRENT.get("best_fitness", -1.0),
                "metrics_history": _CURRENT.get("epoch_metrics", []),
                "final_metrics": _CURRENT["epoch_metrics"][-1] if _CURRENT["epoch_metrics"] else {}
            }, f, indent=2, ensure_ascii=False)
        
        _root_logger.info(f"[TRAIN] 训练完成！已保存最终检查点和完整指标历史")
    except Exception as e:
        _root_logger.error(f"[TRAIN] 训练结束保存失败：{e}")

def _attach_callbacks(model: YOLO):
    """附加训练回调"""
    if int(os.environ.get("RANK", "0")) == 0:
        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_fit_epoch_end", _compact_epoch_line)
        model.add_callback("on_train_end", _on_train_end)

# ============== 应急保存与续训 ==============
def _emergency_save(reason: str = "unknown"):
    """尽最大能力把当前状态落盘到 runs/detect/<name>/weights。"""
    if _CURRENT["emergency_done"]:
        return
    _CURRENT["emergency_done"] = True

    save_dir = _CURRENT.get("save_dir")
    trainer  = _CURRENT.get("trainer")
    model    = _CURRENT.get("model")

    if not save_dir:
        return

    _root_logger.warning(f"[EMERGENCY] 触发应急保存，原因：{reason}")

    # 使用完整保存函数
    try:
        if trainer:
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            _save_checkpoint_with_metrics(
                trainer, epoch, save_dir,
                filename="emergency.pth",
                reason=reason
            )
    except Exception as e:
        _root_logger.error(f"[EMERGENCY] 完整保存失败：{e}")
        
        # 降级：只保存权重
        try:
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            if trainer and hasattr(trainer, "model") and trainer.model is not None:
                torch.save(trainer.model.state_dict(), weights_dir / "emergency.pt")
            elif isinstance(model, YOLO) and hasattr(model, "model") and model.model is not None:
                torch.save(model.model.state_dict(), weights_dir / "emergency.pt")
            
            _root_logger.info("[EMERGENCY] 已保存基本权重文件")
        except Exception as e2:
            _root_logger.error(f"[EMERGENCY] 降级保存也失败：{e2}")

    try:
        with open(save_dir / "EMERGENCY_SAVED.txt", "a", encoding="utf-8") as f:
            f.write(f"[EMERG] reason={reason}\n")
    except Exception:
        pass

def _install_signal_handlers():
    """安装信号处理器"""
    def _handler(signum, frame):
        _emergency_save(reason=f"signal:{signum}")
        # 让进程按常规终止
        os._exit(1)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass
    # 正常退出也兜底
    atexit.register(lambda: _emergency_save(reason="atexit"))

def _maybe_resume_setup(model: YOLO, exp_dir: Path, train_args: Dict[str, Any]) -> None:
    """
    若当前实验目录已存在检查点，尝试恢复训练。
    优先级：
    1. last.pt (Ultralytics 原生)
    2. final_checkpoint.pth (我们的完整检查点)
    3. checkpoint_*.pth (周期性检查点，取最新的)
    4. emergency.pth
    """
    weights_dir = exp_dir / "weights"
    last_pt = weights_dir / "last.pt"
    
    if last_pt.exists():
        # 使用原生 resume 方式（最稳）
        train_args["resume"] = True
        _root_logger.info(f"[RESUME] 检测到 {last_pt}，使用 Ultralytics resume=True 续训。")
        return

    # 查找我们的检查点
    checkpoints = []
    if weights_dir.exists():
        for ckpt_file in weights_dir.glob("*.pth"):
            if ckpt_file.name.startswith("checkpoint_epoch"):
                try:
                    # 从文件名提取 epoch 号
                    epoch_num = int(ckpt_file.stem.split("epoch")[-1])
                    checkpoints.append((epoch_num, ckpt_file))
                except Exception:
                    pass
        
        # 也考虑 final 和 emergency
        if (weights_dir / "final_checkpoint.pth").exists():
            checkpoints.append((999999, weights_dir / "final_checkpoint.pth"))
        if (weights_dir / "emergency.pth").exists():
            checkpoints.append((999998, weights_dir / "emergency.pth"))
    
    if not checkpoints:
        _root_logger.info("[RESUME] 未找到可用的检查点，从头开始训练。")
        return

    # 取最新的检查点
    checkpoints.sort(reverse=True)
    _, ckpt_path = checkpoints[0]
    
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # 恢复模型权重
        if "model_state_dict" in ckpt and hasattr(model, "model") and model.model is not None:
            model.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            _root_logger.info(f"[RESUME] 已加载模型权重：{ckpt_path}")
        
        # 恢复训练历史
        if "epoch_metrics" in ckpt:
            _CURRENT["epoch_metrics"] = ckpt["epoch_metrics"]
            _root_logger.info(f"[RESUME] 已恢复 {len(_CURRENT['epoch_metrics'])} 个 epoch 的指标历史")
        
        if "best_fitness" in ckpt:
            _CURRENT["best_fitness"] = ckpt["best_fitness"]
            _root_logger.info(f"[RESUME] 已恢复最佳 fitness: {_CURRENT['best_fitness']:.4f}")
        
        # 注意：优化器状态无法通过这种方式恢复（需要 Ultralytics 原生 resume）
        # 但模型权重和训练历史已恢复
        
    except Exception as e:
        _root_logger.warning(f"[RESUME] 加载检查点失败：{e}")

# ============== 其它工具 ==============
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _first_cuda_from_device_str(dev_str: str) -> str:
    s = str(dev_str).strip().lower()
    if s == "cpu":
        return "cpu"
    if "," in s:
        return s.split(",")[0].strip()
    return s

# ============== 主流程 ==============
def main():
    _install_signal_handlers()
    set_seed(CFG["train"]["seed"])

    # 1) 数据集准备
    data_dict = prepare_yolo_dataset(CFG["dataset"])
    data_yaml = _ensure_yaml_from_data_dict(data_dict)

    split_summary = Path(data_dict["path"]) / "_split_summary.json"
    if split_summary.exists():
        try:
            ss = json.loads(split_summary.read_text("utf-8"))
            _root_logger.info(f"[DATASET] summary: {json.dumps(ss.get('global', {}), ensure_ascii=False)}")
        except Exception:
            _root_logger.warning("读取 _split_summary.json 失败，跳过。")

    # 2) TEST 池
    data_root = data_dict["path"]
    test_rel  = data_dict["test"]
    test_dir  = test_rel if os.path.isabs(test_rel) else os.path.join(data_root, test_rel)
    all_test_imgs = list_images(test_dir)
    if len(all_test_imgs) == 0:
        raise FileNotFoundError(f"未发现测试集图片：{test_dir}")

    train_device = CFG["train"]["device"]
    eval_device  = _first_cuda_from_device_str(train_device)

    global_rows = []
    global_plot_dir = "runs/_global_plots"; ensure_dir(global_plot_dir)

    # 3) 逐模型训练与评测
    for exp in CFG["experiments"]:
        model_name = exp["model"]
        exp_name   = exp.get("name", Path(model_name).stem)

        # 预期目录（Ultralytics 可能会自动加后缀；我们先把 console 日志打在"预期目录"，后面如有差异再搬家）
        expected_dir = Path(CFG["train"]["project"]) / exp_name
        expected_dir.mkdir(parents=True, exist_ok=True)
        tmp_console_log = expected_dir / "_train_tmp.log"

        # 写 root logger 到预期目录（后续如 save_dir 改变，会转移）
        exp_file_handler_tmp = _add_file_handler_once(_root_logger, expected_dir / "train_log.txt")

        _root_logger.info(f"\n========== [{model_name}] 开始训练（device={train_device}） ==========")

        # 训练阶段：stdout/stderr 只写文件
        with tee_stdout_stderr(tmp_console_log):
            model = YOLO(model_name)
            _CURRENT["model"] = model
            _CURRENT["exp_name"] = exp_name
            _CURRENT["save_dir"] = expected_dir  # 可能会被 on_train_start 更新为真实 save_dir
            _CURRENT["emergency_done"] = False   # 重置应急保存标志

            _attach_callbacks(model)

            train_args = dict(
                data=data_yaml,
                imgsz=CFG["train"]["imgsz"],
                epochs=CFG["train"]["epochs"],
                batch=CFG["train"]["batch"],
                workers=CFG["train"]["workers"],
                device=train_device,
                project=CFG["train"]["project"],
                name=exp_name,
                cos_lr=CFG["train"]["cos_lr"],
                amp=CFG["train"]["amp"],
                exist_ok=True,
                optimizer=CFG["train"]["optimizer"],
                weight_decay=CFG["train"]["weight_decay"],
                cache=CFG["train"]["cache"],
                multi_scale=CFG["train"]["multi_scale"],
                degrees=CFG["train"]["degrees"],
                hsv_h=CFG["train"]["hsv_h"], hsv_s=CFG["train"]["hsv_s"], hsv_v=CFG["train"]["hsv_v"],
                patience=CFG["train"]["patience"],
                verbose=False,
            )

            # 如果该实验目录已有检查点，尝试续训/预载权重
            _maybe_resume_setup(model, expected_dir, train_args)

            try:
                results = model.train(**train_args)
            except Exception as e:
                # 发生异常，尽力把当前状态落盘
                _root_logger.error(f"[EXCEPTION] 训练中断：{e}\n{traceback.format_exc()}")
                _emergency_save(reason="exception")
                raise

        # ====== 更稳妥地获取 save_dir ======
        trainer_obj = getattr(model, "trainer", None)
        save_dir = (
            Path(getattr(results, "save_dir", "")) if results and getattr(results, "save_dir", None) else
            Path(getattr(trainer_obj, "save_dir", "")) if trainer_obj and getattr(trainer_obj, "save_dir", None) else
            Path(_CURRENT.get("save_dir") or expected_dir)
        )
        _CURRENT["save_dir"] = save_dir  # 确认最终目录

        # 把临时 console 日志移动到最终目录
        final_console_log = save_dir / "train_console.log"
        try:
            final_console_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_console_log), str(final_console_log))
        except Exception:
            try:
                shutil.copy2(str(tmp_console_log), str(final_console_log))
            except Exception:
                _root_logger.warning(f"移动/复制 console 日志失败：{tmp_console_log} -> {final_console_log}")

        # 把 root logger 的文件 handler 也移到最终目录（重新绑定）
        _remove_handler(_root_logger, exp_file_handler_tmp)
        exp_file_handler = _add_file_handler_once(_root_logger, save_dir / "train_log.txt")

        # ★ 优先使用我们保存的最佳模型
        best_fitness_pt = str(save_dir / "weights" / "best_fitness.pt")
        best_pt = best_fitness_pt if Path(best_fitness_pt).exists() else str(save_dir / "weights" / "best.pt")
        last_pt = str(save_dir / "weights" / "last.pt")

        _root_logger.info(f"使用最佳模型进行评估：{best_pt}")

        # 验证
        _root_logger.info(f"验证中（COCO 协议，device={eval_device}）...")
        with tee_stdout_stderr(final_console_log):
            model_best = YOLO(best_pt)
            val_res = model_best.val(
                data=data_yaml,
                imgsz=CFG["train"]["imgsz"],
                device=eval_device,
                plots=True,
                rect=True,
                save_json=True
            )

        m = extract_ultralytics_metrics(val_res)
        size_mb = file_size_mb(best_pt)

        # 时延
        rng = np.random.default_rng(3)
        pool = list(rng.choice(all_test_imgs, size=min(len(all_test_imgs), CFG["latency"]["pool_samples"]), replace=False))
        _root_logger.info(f"时延统计（TEST）：pool={len(pool)}, runs={CFG['latency']['runs']} (device={eval_device})")
        with tee_stdout_stderr(final_console_log):
            latency = compute_latency(
                model_best, pool, CFG["train"]["imgsz"], eval_device,
                warmup=CFG["latency"]["warmup"], runs=CFG["latency"]["runs"]
            )

        # 训练曲线（直接用 metrics_95 的实现）
        results_csv = str(save_dir / "results.csv")
        plots_dir   = str(save_dir / "_plots")
        try:
            plot_training_curves(results_csv, plots_dir)
            _root_logger.info(f"训练曲线已保存：{plots_dir}")
        except Exception as e:
            _root_logger.warning(f"训练曲线绘制失败：{e}")

        # 可视化预测（可选）
        if CFG["predict_vis"]["enable"]:
            vis_n = min(CFG["predict_vis"]["num_images"], len(pool))
            subset = pool[:vis_n]
            with tee_stdout_stderr(final_console_log):
                model_best.predict(
                    source=subset,
                    imgsz=CFG["train"]["imgsz"],
                    device=eval_device,
                    save=True, conf=CFG["predict_vis"]["conf"],
                    project=str(save_dir), name="_pred_vis", exist_ok=True, verbose=False
                )
            _root_logger.info(f"预测可视化已输出：{save_dir / '_pred_vis'}")

        # 汇总
        summary = {
            "model": model_name,
            "exp_name": exp_name,
            "weights": best_pt,
            "imgsz": CFG["train"]["imgsz"],
            "epochs": CFG["train"]["epochs"],
            "batch": CFG["train"]["batch"],
            "device": train_device,
            "best_fitness": float(_CURRENT.get("best_fitness", -1.0)),
            "metrics": {
                "mAP@0.5": m["map50"],
                "mAP@0.75": m["map75"],        # ★ 新增
                "mAP@0.95": m["map95"],        # ★ 新增
                "mAP@[0.5:0.95]": m["map5095"],
                "precision": m["precision"],
                "recall": m["recall"]
            },
            "latency_ms": {
                "mean": latency["latency_mean_ms"],
                "p95": latency["latency_p95_ms"],
                "runs": latency["runs"]
            },
            "size_mb": size_mb,
            "artifacts": {
                "results_csv": results_csv,
                "plots_dir": plots_dir,
                "best_pt": best_pt,
                "last_pt": last_pt,
                "console_log": str(final_console_log),
                "train_log": str(save_dir / "train_log.txt"),
                "metrics_history": str(save_dir / "training_metrics_complete.json"),
                "epoch_compact_log": str(save_dir / "epoch_compact.log"),
                "epoch_detailed_log": str(save_dir / "epoch_detailed.jsonl")
            }
        }
        save_summary_json(str(save_dir / "metrics_summary.json"), summary)

        latex_row = latex_row_for(summary, f"YOLOv11({Path(model_name).stem})")
        with open(save_dir / "latex_row.txt", "w", encoding="utf-8") as f:
            f.write(latex_row)

        _root_logger.info("=== SUMMARY ===")
        _root_logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
        _root_logger.info("LaTeX row:")
        _root_logger.info(latex_row)
        _root_logger.info(f"Artifacts in: {save_dir}")

        global_rows.append({
            "model": f"YOLOv11-{Path(model_name).stem}",
            "mAP@[0.5:0.95]": summary["metrics"]["mAP@[0.5:0.95]"],
            "mAP@0.5": summary["metrics"]["mAP@0.5"],
            "latency_mean_ms": summary["latency_ms"]["mean"],
            "latency_p95_ms": summary["latency_ms"]["p95"],
            "size_mb": summary["size_mb"],
            "save_dir": str(save_dir)
        })

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        _remove_handler(_root_logger, exp_file_handler)

    # 4) 全局对比图与 CSV
    if global_rows:
        df = pd.DataFrame(global_rows)
        global_plot_dir = "runs/_global_plots"; ensure_dir(global_plot_dir)
        df_path = os.path.join(global_plot_dir, "summary.csv")
        df.to_csv(df_path, index=False, encoding="utf-8-sig")
        plot_global_bars(df, global_plot_dir)

        latex_all_path = os.path.join(global_plot_dir, "latex_rows_all.txt")
        with open(latex_all_path, "w", encoding="utf-8") as f:
            for row in global_rows:
                f.write(
                    f"YOLOv11({row['model'].split('-')[-1]}) & "
                    f"{row['mAP@[0.5:0.95]']:.2f} & {row['mAP@0.5']:.2f} & "
                    f"{row['latency_mean_ms']:.2f} & {row['latency_p95_ms']:.2f} & "
                    f"{row['size_mb']:.2f} \\\\\n"
                )

        _root_logger.info(f"[GLOBAL] 汇总 CSV：{df_path}")
        _root_logger.info(f"[GLOBAL] 图表目录：{global_plot_dir}/*.png")
        _root_logger.info(f"[GLOBAL] LaTeX 行：{latex_all_path}")

if __name__ == "__main__":
    main()
