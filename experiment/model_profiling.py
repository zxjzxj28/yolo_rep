"""Utility for downloading and profiling common detection models.

This script downloads the checkpoints used in the accompanying experiments and
reports each model's parameter count and an approximate FLOPs estimate using a
dummy forward pass. It is designed to make it easy to reproduce the model
card-style summary table in the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from thop import profile
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from ultralytics import YOLO


@dataclass
class ModelSpec:
    """Configuration for a single model profile run."""

    name: str
    weight: str
    builder: Callable[[str], torch.nn.Module]
    input_size: Tuple[int, int]
    expects_list_inputs: bool = False


def _load_ultralytics_model(weight: str, device: str) -> torch.nn.Module:
    model = YOLO(weight).model
    model.to(device)
    return model


def _load_faster_rcnn(device: str) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    return model


def _load_ssd_mobilenet(device: str) -> torch.nn.Module:
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )
    model.to(device)
    return model


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="YOLO11-Small",
        weight="yolo11s.pt",
        builder=_load_ultralytics_model,
        input_size=(640, 640),
    ),
    ModelSpec(
        name="YOLO11-Nano",
        weight="yolo11n.pt",
        builder=_load_ultralytics_model,
        input_size=(640, 640),
    ),
    ModelSpec(
        name="YOLOv8-Small",
        weight="yolov8s.pt",
        builder=_load_ultralytics_model,
        input_size=(640, 640),
    ),
    ModelSpec(
        name="YOLOv8-Nano",
        weight="yolov8n.pt",
        builder=_load_ultralytics_model,
        input_size=(640, 640),
    ),
    ModelSpec(
        name="Faster R-CNN",
        weight="fasterrcnn_resnet50_fpn",
        builder=_load_faster_rcnn,
        input_size=(640, 640),
        expects_list_inputs=True,
    ),
    ModelSpec(
        name="SSD-MobileNetV3-L",
        weight="ssdlite320_mobilenet_v3_large",
        builder=_load_ssd_mobilenet,
        input_size=(320, 320),
        expects_list_inputs=True,
    ),
    ModelSpec(
        name="RT-DETR-Nano",
        weight="rtdetr-l.pt",
        builder=_load_ultralytics_model,
        input_size=(640, 640),
    ),
]


def _format_rows(rows: Iterable[Tuple[str, float, float]]) -> str:
    header = f"{'Model':25} | {'Params (M)':>10} | {'FLOPs (B)':>10}"
    divider = "-" * len(header)
    row_lines = [
        f"{name:25} | {params:10.2f} | {flops:10.2f}" for name, params, flops in rows
    ]
    return "\n".join([header, divider, *row_lines])


def _profile_model(spec: ModelSpec, device: str, img_size: Tuple[int, int]) -> Tuple[float, float]:
    model = spec.builder(device)
    model.eval()

    dummy = torch.randn(1, 3, img_size[0], img_size[1], device=device)
    inputs = ([dummy],) if spec.expects_list_inputs else (dummy,)

    with torch.no_grad():
        macs, params = profile(model, inputs=inputs, verbose=False)

    params_m = params / 1e6
    flops_b = macs / 1e9
    return params_m, flops_b


def run(
    models: List[str], device: str, override_size: Optional[int]
) -> List[Tuple[str, float, float]]:
    lookup = {spec.name.lower(): spec for spec in MODEL_SPECS}
    selected_specs = MODEL_SPECS if "all" in models else [lookup[m.lower()] for m in models]

    rows = []
    for spec in selected_specs:
        img_h, img_w = spec.input_size
        if override_size:
            img_h = img_w = override_size

        print(f"Profiling {spec.name} on {device} with input {img_h}x{img_w} ...")
        params, flops = _profile_model(spec, device, (img_h, img_w))
        rows.append((spec.name, params, flops))

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and profile YOLO-family, Faster R-CNN, SSD, and RT-DETR models "
            "for parameter count and FLOPs."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", *[spec.name for spec in MODEL_SPECS]],
        help="Subset of models to profile. Defaults to all entries in the experiment table.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to profile on (cpu or cuda). Ensure the device is available before using it.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help=(
            "Override the square input resolution for all models. By default the image size "
            "matches the configuration used in the experiment table."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the results as a CSV file.",
    )
    return parser.parse_args()


def _save_csv(rows: List[Tuple[str, float, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("model,params_m,flops_b\n")
        for name, params, flops in rows:
            file.write(f"{name},{params:.4f},{flops:.4f}\n")
    print(f"Saved results to {output_path}")


def main() -> None:
    args = _parse_args()
    rows = run(args.models, args.device, args.img_size)

    print("\n" + _format_rows(rows))
    if args.output:
        _save_csv(rows, args.output)


if __name__ == "__main__":
    main()
