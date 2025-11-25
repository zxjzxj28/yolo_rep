# -*- coding: utf-8 -*-
# dataset_io.py  （多目录：每个目录内按比例分，再整合到统一 YOLO 结构；支持分配复用）
import os, glob, shutil, random, json, math, csv
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

    # ===== 固定分配并复用 =====
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

# ---------- 新增：严格“桶→比例切分”策略 ----------
def _split_list_by_ratio(idxs: List[int], ratios: Tuple[float, float, float], rng: random.Random):
    """
    将一个索引列表按给定比例切分为 (train, val, test)，使用“最大余数法”保证总数精确。
    """
    n = len(idxs)
    if n == 0:
        return [], [], []
    idxs = idxs.copy()
    rng.shuffle(idxs)

    t_r, v_r, e_r = ratios
    raw = [t_r * n, v_r * n, e_r * n]
    base = [math.floor(x) for x in raw]
    rem = n - sum(base)
    # 按小数部分从大到小分配余数
    fracs = sorted([(raw[i] - base[i], i) for i in range(3)], reverse=True)
    for k in range(rem):
        base[fracs[k][1]] += 1
    t_sz, v_sz, e_sz = base
    return idxs[:t_sz], idxs[t_sz:t_sz + v_sz], idxs[t_sz + v_sz:]

def _ratio_split_by_neg_and_classes(
    n_items: int,
    cls_sets: List[Set[int]],
    ratios: Tuple[float, float, float],
    seed: int,
    num_classes: int = 3
):
    """
    按“桶→比例切分”的策略返回 (train_idx, val_idx, test_idx)
    桶包括：
      - neg：无标签
      - single_c：仅包含类别 c (c in [0..num_classes-1])
      - multi：包含 >=2 个类别 或 超出 num_classes 的类别
    负样本与每个单类正样本各自按比例切分；multi 单独按比例切分。
    """
    rng = random.Random(seed)

    neg = []
    single = {c: [] for c in range(num_classes)}
    multi = []

    for i, s in enumerate(cls_sets):
        if not s:
            neg.append(i)
        elif len(s) == 1:
            c = next(iter(s))
            if 0 <= c < num_classes:
                single[c].append(i)
            else:
                multi.append(i)
        else:
            multi.append(i)

    # 按比例切分并汇总
    tr, va, te = [], [], []

    def apply_bucket(bucket: List[int], seed_offset: int):
        t, v, e = _split_list_by_ratio(bucket, ratios, random.Random(seed + seed_offset))
        tr.extend(t); va.extend(v); te.extend(e)

    # 负样本
    apply_bucket(neg, 11)

    # 三种单类正样本
    for c in range(num_classes):
        apply_bucket(single[c], 100 + c)

    # 多类样本（解耦处理）
    apply_bucket(multi, 999)

    # 质量检查
    all_idx = set(tr) | set(va) | set(te)
    assert len(all_idx) == n_items, f"切分总量不一致: {n_items} vs {len(all_idx)}"
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "切分重复分配"

    # 返回及便于后续统计的“桶构成”
    buckets = {
        "neg": neg,
        "single": single,  # dict: c -> [idx...]
        "multi": multi
    }
    return tr, va, te, buckets

# ---------- 统计/持久化工具 ----------
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

def _split_stats(pairs: List[Tuple[Path, Optional[Path]]], num_classes: int):
    """
    返回该 split 的详细统计：
      - total、neg（无标签数）、multi（多类样本数）
      - single_only[c]：仅含类 c 的样本数
      - contains_class[c]：包含类 c 的样本数（multi 也计入）
      - proportions：以上各项 / total
    """
    total = len(pairs)
    neg = 0
    multi = 0
    single_only = {c: 0 for c in range(num_classes)}
    contains = {c: 0 for c in range(num_classes)}

    for (ip, lp) in pairs:
        s = _read_classes_from_label(lp)
        if not s:
            neg += 1
            continue
        if len(s) == 1:
            c = next(iter(s))
            if 0 <= c < num_classes:
                single_only[c] += 1
        else:
            multi += 1
        # 包含统计（multi/单类都计入）
        for c in s:
            if 0 <= c < num_classes:
                contains[c] += 1

    def _props(v):  # 比例（占 total）
        if total == 0: return 0.0
        return round(v / total, 6)

    proportions = {
        "neg": _props(neg),
        "multi": _props(multi),
        "single_only": {c: _props(single_only[c]) for c in range(num_classes)},
        "contains_class": {c: _props(contains[c]) for c in range(num_classes)},
    }
    return {
        "count": total,
        "neg": neg,
        "multi": multi,
        "single_only": single_only,
        "contains_class": contains,
        "proportions": proportions
    }

def _save_ratios_csv(out_root: Path, summary_obj: Dict, class_names: Dict[int, str]):
    """
    将 train/val/test 的计数与比例导出为 CSV：_split_ratios.csv
    列包括：
      split, total, neg_count, neg_prop, multi_count, multi_prop,
      single_only_{name}_count/prop..., contains_{name}_count/prop...
    """
    csv_path = out_root / "_split_ratios.csv"
    num_classes = len(class_names)
    fieldnames = [
        "split", "total",
        "neg_count", "neg_prop",
        "multi_count", "multi_prop",
    ]
    for c in range(num_classes):
        name = class_names.get(c, f"class{c}")
        fieldnames += [f"single_only_{name}_count", f"single_only_{name}_prop"]
    for c in range(num_classes):
        name = class_names.get(c, f"class{c}")
        fieldnames += [f"contains_{name}_count", f"contains_{name}_prop"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ["train", "val", "test"]:
            st = summary_obj["global"][split]
            row = {
                "split": split,
                "total": st["count"],
                "neg_count": st["neg"],
                "neg_prop": st["proportions"]["neg"],
                "multi_count": st["multi"],
                "multi_prop": st["proportions"]["multi"],
            }
            for c in range(num_classes):
                name = class_names.get(c, f"class{c}")
                row[f"single_only_{name}_count"] = st["single_only"].get(str(c), st["single_only"].get(c, 0))
                row[f"single_only_{name}_prop"]  = st["proportions"]["single_only"].get(str(c), st["proportions"]["single_only"].get(c, 0.0))
            for c in range(num_classes):
                name = class_names.get(c, f"class{c}")
                row[f"contains_{name}_count"] = st["contains_class"].get(str(c), st["contains_class"].get(c, 0))
                row[f"contains_{name}_prop"]  = st["proportions"]["contains_class"].get(str(c), st["proportions"]["contains_class"].get(c, 0.0))
            writer.writerow(row)

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

    # 类别数
    if isinstance(cfg.get("names"), dict):
        num_classes = len(cfg["names"])
        class_names = {int(k): v for k, v in cfg["names"].items()}
    else:
        num_classes = 3
        class_names = {0: "class0", 1: "class1", 2: "class2"}

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
        summary_global = {
            "train": _split_stats(merged.get("train", []), num_classes),
            "val":   _split_stats(merged.get("val",   []), num_classes),
            "test":  _split_stats(merged.get("test",  []), num_classes),
        }
        # 将 dict 的 key 统一转为字符串，便于一致的 JSON
        def _stringify_keys(d):
            if isinstance(d, dict):
                return {str(k): _stringify_keys(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [_stringify_keys(x) for x in d]
            return d

        summary = {
            "per_root": "(reused manifest)",
            "global": _stringify_keys(summary_global),
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
                    "names": class_names
                }, f, allow_unicode=True, sort_keys=False)
        except Exception:
            pass

        with open(out_root / "_split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 导出 CSV 比例表
        _save_ratios_csv(out_root, summary, class_names)

        print(f"[OK] 已按清单复用分配：{manifest_path}")
        return {
            "path": str(out_root),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "names": class_names
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

        # =========== 关键修改：按负样本 + 单类正样本 + 多类桶 比例切分 ===========
        ti, vi, ei, _bucket_dbg = _ratio_split_by_neg_and_classes(
            n_items=len(imgs),
            cls_sets=cls_sets,
            ratios=ratios,
            seed=seed_i,
            num_classes=num_classes
        )

        # 目录级统计（方便确认上游数据本身的分布）
        def _stat(indices):
            total = len(indices)
            neg = 0
            multi = 0
            single = {c: 0 for c in range(num_classes)}
            contains = {c: 0 for c in range(num_classes)}
            for i in indices:
                s = cls_sets[i]
                if not s:
                    neg += 1
                elif len(s) == 1:
                    c = next(iter(s))
                    if 0 <= c < num_classes:
                        single[c] += 1
                else:
                    multi += 1
                for c in s:
                    if 0 <= c < num_classes:
                        contains[c] += 1
            return {"count": total, "neg": neg, "multi": multi, "single_only": single, "contains_class": contains}

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

    # 汇总统计（全局）
    def _global_stat(pairs):
        return _split_stats(pairs, num_classes)

    global_summary = {
        "train": _global_stat(merged_pairs["train"]),
        "val":   _global_stat(merged_pairs["val"]),
        "test":  _global_stat(merged_pairs["test"]),
    }

    summary = {
        "per_root": per_root_stats,
        "global": global_summary,
        "ratios": ratios,
        "mode": mode,
        "roots": {"images": image_dirs, "labels": label_dirs},
        "reused": False,
        "class_names": class_names
    }

    if cfg.get("dry_run", False):
        print("[DRY RUN] 仅统计，不写入：", json.dumps(summary, ensure_ascii=False, indent=2))
        return {
            "path": str(out_root),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "names": class_names
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

    # 写 data.yaml、summary、manifest
    try:
        import yaml
        with open(out_root / "mobilechart.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "path": str(out_root),
                "train": "images/train",
                "val":   "images/val",
                "test":  "images/test",
                "names": class_names
            }, f, allow_unicode=True, sort_keys=False)
    except Exception:
        pass

    # 统一把 dict 的 int key 转成 str，便于 JSON 稳定展示
    def _stringify_keys(d):
        if isinstance(d, dict):
            return {str(k): _stringify_keys(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [_stringify_keys(x) for x in d]
        return d

    summary_for_json = _stringify_keys(summary)
    with open(out_root / "_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_for_json, f, ensure_ascii=False, indent=2)

    # 保存清单，供后续复用
    _save_manifest(manifest_path, merged_pairs, cfg)

    # 导出 CSV 比例表
    _save_ratios_csv(out_root, summary_for_json, class_names)

    print(f"[OK] 数据集已准备完毕：{out_root}")
    print(f"[OK] 分配清单已保存：{manifest_path}")
    print(f"[OK] 分配比例已保存：{out_root / '_split_summary.json'}  与  {out_root / '_split_ratios.csv'}")

    return {
        "path": str(out_root),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": class_names
    }

# 直接运行调试
if __name__ == "__main__":
    prepare_yolo_dataset(DATASET_CFG)
