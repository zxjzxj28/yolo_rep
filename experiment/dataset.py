# -*- coding: utf-8 -*-
# dataset_io.py  （多目录：每个目录内按比例分，再整合到统一 YOLO 结构；支持分配复用）
import os, glob, shutil, random, json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# ============== 配置（只改这里） ==============
DATASET_CFG = {
    # 多个图片目录（只取 *.png，可按需改模式为递归）
    "image_dirs": [
        "/graph/zxj/detection/fused_img",
        "/graph/zxj/detection/neg_data/image",
        "/graph/zxj/zh/images"
    ],
    # 多个标签目录（只取 *.txt），按同文件名（不含后缀）对应
    "label_dirs": [
        "/graph/zxj/detection/labels",
        "/graph/zxj/detection/neg_data/labels",
        "/graph/zxj/zh/labels"
    ],
    "out_root":   "/graph/zxj/mobileChart",  # 输出根目录（images/labels/{train,val,test}）
    "split":      (0.6, 0.2, 0.2),           # 6:2:2
    "seed":       42,
    "names": {0: "bar", 1: "line", 2: "pie"},
    "mode": "symlink",                       # "copy" | "move" | "symlink"
    "dry_run": False,
    "recursive_images": False,               # True=递归扫描图片目录

    # ===== 新增：固定分配并复用 =====
    "reuse_existing_split": True,            # 若 True 且清单存在，则复用，不再重新切分
    "clean_out": True,                       # 写入前清空 out_root 下 images/labels，避免残留
    "split_manifest": None                   # 自定义清单路径；默认 {out_root}/_split_manifest.json
}
# ============================================

def _assert_dirs_exist(dirs: List[str], kind: str):
    for d in dirs:
        if not Path(d).exists():
            raise FileNotFoundError(f"{kind} 目录不存在：{d}")

def _list_pngs(dirp: str, recursive=False) -> List[Path]:
    if not recursive:
        return sorted([Path(p) for p in glob.glob(os.path.join(dirp, "*.png"))])
    # 递归
    root = Path(dirp)
    files = list(root.rglob("*.png")) + list(root.rglob("*.PNG"))
    return sorted([p for p in files if p.is_file()])

def _collect_txts_multi_unique(label_dirs: List[str]) -> Dict[str, Path]:
    """
    从多个目录收集 *.txt 标签，返回 {stem -> Path}。
    假设不会有重名（stem 唯一），若发现重名则抛错。
    """
    index: Dict[str, Path] = {}
    for root in label_dirs:
        for p in glob.glob(os.path.join(root, "*.txt")):
            pp = Path(p); stem = pp.stem
            if stem in index:
                raise ValueError(f"发现重复标签文件名（不含后缀）：{stem}\n"
                                 f" - {index[stem]}\n - {pp}\n"
                                 f"请确保多目录中标签名唯一，或先做合并去重。")
            index[stem] = pp
    return index

def _read_classes_from_label(lbl_path: Optional[Path]) -> Set[int]:
    if (lbl_path is None) or (not Path(lbl_path).exists()):
        return set()
    classes = set()
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                cid = int(parts[0]); classes.add(cid)
            except Exception:
                pass
    return classes

def _mkdirs(root: Path):
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

def _place(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)
    else:
        raise ValueError("mode must be copy|move|symlink")

def _greedy_multilabel_stratified_split(items, cls_sets, ratios, seed=42):
    """
    近似多标签分层：优先分配“稀有类多”的样本到对该类更缺的 split。
    返回：train_idx, val_idx, test_idx
    """
    random.seed(seed)
    n = len(items)
    t_sz = int(n * ratios[0]); v_sz = int(n * ratios[1]); e_sz = n - t_sz - v_sz

    from collections import Counter
    freq = Counter()
    for s in cls_sets:
        for c in s: freq[c] += 1

    def rarity_score(s):
        if not s: return 0.0
        return sum(1.0 / max(1, freq[c]) for c in s)

    order = list(range(n))
    order.sort(key=lambda i: (rarity_score(cls_sets[i]), len(cls_sets[i])), reverse=True)

    split_cls_cnt = [Counter(), Counter(), Counter()]
    split_bins = [[], [], []]

    def pick_split(s):
        caps = [t_sz, v_sz, e_sz]
        sizes = [len(split_bins[0]), len(split_bins[1]), len(split_bins[2])]
        candidates = [i for i in range(3) if sizes[i] < caps[i]]
        if not candidates:
            return random.randint(0, 2)
        if not s:
            return min(candidates, key=lambda i: sizes[i])
        def need_score(i):
            return sum(-split_cls_cnt[i][c] for c in s)
        return min(candidates, key=need_score)

    for i in order:
        sp = pick_split(cls_sets[i])
        split_bins[sp].append(i)
        for c in cls_sets[i]:
            split_cls_cnt[sp][c] += 1

    return split_bins[0], split_bins[1], split_bins[2]

# ===== 新增：清空输出目录、保存/读取分配清单 =====
def _clean_out_dirs(out_root: Path):
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        p = out_root / sub
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

def _save_manifest(manifest_path: Path, merged_pairs: Dict[str, List[Tuple[Path, Optional[Path]]]], cfg: Dict):
    manifest = {
        "mode": cfg["mode"],
        "ratios": cfg["split"],
        "seed": cfg["seed"],
        "roots": {"images": cfg["image_dirs"], "labels": cfg["label_dirs"]},
        "splits": {
            sp: [{"img": str(ip), "lbl": (str(lp) if lp is not None else None)} for (ip, lp) in pairs]
            for sp, pairs in merged_pairs.items()
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def _load_manifest(manifest_path: Path) -> Dict[str, List[Tuple[Path, Optional[Path]]]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    splits = {}
    for sp, items in m["splits"].items():
        pairs = []
        for it in items:
            ip = Path(it["img"])
            lp = Path(it["lbl"]) if it.get("lbl") else None
            pairs.append((ip, lp))
        splits[sp] = pairs
    return splits

def prepare_yolo_dataset(cfg: Dict) -> Dict:
    """
    对每个图片目录分别按比例切分（保留各自分布），再合并导出 YOLO 结构：
    {path, train, val, test, names}
    对齐规则：按同文件名（不含后缀）在 label_dirs 中查找标签；无则视为负样本。
    若开启 reuse_existing_split 且清单存在，则直接按清单复用，不再重新切分。
    """
    image_dirs: List[str] = cfg["image_dirs"]
    label_dirs: List[str] = cfg["label_dirs"]
    out_root = Path(cfg["out_root"])
    ratios   = cfg["split"]
    mode     = cfg["mode"]
    recursive= cfg.get("recursive_images", False)

    if not image_dirs or not label_dirs:
        raise ValueError("image_dirs / label_dirs 不能为空列表")
    _assert_dirs_exist(image_dirs, "图片")
    _assert_dirs_exist(label_dirs, "标签")

    # 清单路径与复用开关
    manifest_path = Path(cfg.get("split_manifest") or (out_root / "_split_manifest.json"))
    reuse = bool(cfg.get("reuse_existing_split", False)) and manifest_path.exists()

    # ===== 复用清单分配（不重新切分）=====
    if reuse:
        if mode == "move":
            raise ValueError("复用清单与 mode='move' 冲突：源文件上次已被移走，无法复现。请改为 'symlink' 或 'copy'。")
        if cfg.get("clean_out", False) and out_root.exists():
            _clean_out_dirs(out_root)
        _mkdirs(out_root)

        merged = _load_manifest(manifest_path)  # {"train":[(img,label),...], ...}

        # 直接落盘（复现）
        def _dump_reuse(pairs, split):
            for (ip, lp) in pairs:
                if not Path(ip).exists():
                    print(f"[WARN] 清单中的图片缺失，跳过：{ip}")
                    continue
                ip_dst = out_root / "images" / split / Path(ip).name
                lp_dst = out_root / "labels" / split / (Path(ip).stem + ".txt")
                _place(Path(ip), ip_dst, mode)
                if lp is not None and Path(lp).exists():
                    _place(Path(lp), lp_dst, mode)
                else:
                    lp_dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(lp_dst, "w", encoding="utf-8") as f:
                        f.write("")

        _dump_reuse(merged.get("train", []), "train")
        _dump_reuse(merged.get("val",   []), "val")
        _dump_reuse(merged.get("test",  []), "test")

        # 汇总统计（基于源标签路径）
        def _global_stat(pairs):
            from collections import Counter
            total = len(pairs); c = Counter(); neg = 0
            for (ip, lp) in pairs:
                s = _read_classes_from_label(lp)
                if not s: neg += 1
                for cc in s: c[cc] += 1
            return {"count": total, "neg": neg, "class_freq": dict(c)}

        summary = {
            "per_root": "(reused manifest)",
            "global": {
                "train": _global_stat(merged.get("train", [])),
                "val":   _global_stat(merged.get("val",   [])),
                "test":  _global_stat(merged.get("test",  []))
            },
            "ratios": ratios,
            "mode": mode,
            "roots": {"images": image_dirs, "labels": label_dirs},
            "manifest": str(manifest_path),
            "reused": True
        }

        # 写 data.yaml 和 summary
        try:
            import yaml
            with open(out_root / "mobilechart.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "path": str(out_root),
                    "train": "images/train",
                    "val":   "images/val",
                    "test":  "images/test",
                    "names": cfg["names"]
                }, f, allow_unicode=True, sort_keys=False)
        except Exception:
            pass

        with open(out_root / "_split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[OK] 已按清单复用分配：{manifest_path}")
        return {
            "path": str(out_root),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "names": cfg["names"]
        }

    # ===== 首次/重建：正常切分并生成清单 =====
    # 标签全集索引
    lbl_map = _collect_txts_multi_unique(label_dirs)  # {stem: Path}

    # 全局集合，用于检测输出文件名冲突
    global_seen_imgnames = set()

    # 汇总三份 split 的 (img_path, lbl_path) 列表
    merged_pairs: Dict[str, List[Tuple[Path, Optional[Path]]]] = {"train": [], "val": [], "test": []}

    # 每目录统计
    per_root_stats = {}

    # 遍历每个图片目录：单独切分
    for ridx, img_root in enumerate(image_dirs):
        imgs = _list_pngs(img_root, recursive=recursive)
        if not imgs:
            print(f"[WARN] 目录无 PNG：{img_root}")
            continue

        stems = [p.stem for p in imgs]
        lbls  = [lbl_map.get(s, None) for s in stems]
        cls_sets = [_read_classes_from_label(lp) for lp in lbls]

        # 为了可复现但又避免目录间同一个 seed，给每个目录 seed 做偏移
        seed_i = (cfg["seed"] + 97 * (ridx + 1)) & 0x7fffffff
        ti, vi, ei = _greedy_multilabel_stratified_split(list(range(len(imgs))), cls_sets, ratios, seed=seed_i)

        def _stat(indices):
            from collections import Counter
            total = len(indices); c = Counter(); neg = 0
            for i in indices:
                s = cls_sets[i]
                if not s: neg += 1
                for cc in s: c[cc] += 1
            return {"count": total, "neg": neg, "class_freq": dict(c)}

        per_root_stats[str(img_root)] = {
            "total": len(imgs),
            "train": _stat(ti),
            "val":   _stat(vi),
            "test":  _stat(ei)
        }

        # 加入全局桶（先检查文件名冲突）
        for split, idxs in [("train", ti), ("val", vi), ("test", ei)]:
            for i in idxs:
                ip = imgs[i]
                if ip.name in global_seen_imgnames:
                    raise ValueError(f"检测到跨目录输出文件名冲突：{ip.name}\n"
                                     f"发生在目录：{img_root}\n"
                                     f"请先重命名确保各目录间文件名唯一。")
                global_seen_imgnames.add(ip.name)
                merged_pairs[split].append((ip, lbls[i]))

    # 汇总统计
    def _global_stat(pairs):
        from collections import Counter
        total = len(pairs); c = Counter(); neg = 0
        for (ip, lp) in pairs:
            s = _read_classes_from_label(lp)
            if not s: neg += 1
            for cc in s: c[cc] += 1
        return {"count": total, "neg": neg, "class_freq": dict(c)}

    summary = {
        "per_root": per_root_stats,
        "global": {
            "train": _global_stat(merged_pairs["train"]),
            "val":   _global_stat(merged_pairs["val"]),
            "test":  _global_stat(merged_pairs["test"])
        },
        "ratios": ratios,
        "mode": mode,
        "roots": {"images": image_dirs, "labels": label_dirs},
        "reused": False
    }

    if cfg.get("dry_run", False):
        print("[DRY RUN] 仅统计，不写入：", json.dumps(summary, ensure_ascii=False, indent=2))
        return {
            "path": str(out_root),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "names": cfg["names"]
        }

    # 落盘：可选清空 + 创建
    if cfg.get("clean_out", False) and out_root.exists():
        _clean_out_dirs(out_root)
    _mkdirs(out_root)

    def _dump(pairs, split):
        for (ip, lp) in pairs:
            ip_dst = out_root / "images" / split / ip.name
            lp_dst = out_root / "labels" / split / (ip.stem + ".txt")
            _place(ip, ip_dst, mode)
            if lp is not None and Path(lp).exists():
                _place(Path(lp), lp_dst, mode)
            else:
                lp_dst.parent.mkdir(parents=True, exist_ok=True)
                with open(lp_dst, "w", encoding="utf-8") as f:
                    f.write("")

    _dump(merged_pairs["train"], "train")
    _dump(merged_pairs["val"],   "val")
    _dump(merged_pairs["test"],  "test")

    # 写 data.yaml 和 summary
    try:
        import yaml
        with open(out_root / "mobilechart.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "path": str(out_root),
                "train": "images/train",
                "val":   "images/val",
                "test":  "images/test",
                "names": cfg["names"]
            }, f, allow_unicode=True, sort_keys=False)
    except Exception:
        pass

    with open(out_root / "_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存清单，供后续复用
    _save_manifest(manifest_path, merged_pairs, cfg)
    print(f"[OK] 数据集已准备完毕：{out_root}")
    print(f"[OK] 分配清单已保存：{manifest_path}")

    return {
        "path": str(out_root),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": cfg["names"]
    }

# 直接运行调试
if __name__ == "__main__":
    prepare_yolo_dataset(DATASET_CFG)
