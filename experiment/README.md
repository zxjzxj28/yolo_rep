# 模型参数与 FLOPs 统计工具

`model_profiling.py` 脚本用于一键下载实验中用到的检测模型（YOLO11/YOLOv8 系列、Faster R-CNN、SSD-MobileNetV3-L、RT-DETR）并统计其参数量和 FLOPs，帮助快速复现表 \ref{tab:result} 中的模型能力对比。

## 环境准备

1. 建议使用 Python 3.10+。
2. 安装依赖（包含 PyTorch、TorchVision、Ultralytics 及 FLOPs 计算库 `thop`）：

   ```bash
   pip install -r requirements.txt
   ```

   > 说明：`requirements.txt` 使用官方推理权重，脚本会在首次运行时自动下载对应的模型文件。请确保网络可访问官方模型仓库。

## 快速使用

在 `experiment` 目录下执行：

```bash
python model_profiling.py
```

默认行为：

- 依次下载并加载 YOLO11-Small、YOLO11-Nano、YOLOv8-Small、YOLOv8-Nano、Faster R-CNN、SSD-MobileNetV3-L、RT-DETR-Nano 七个模型。
- 使用各模型默认输入分辨率（YOLO/RT-DETR/Faster R-CNN 采用 640×640，SSD-MobileNetV3-L 采用 320×320）。
- 输出包含参数量（百万级）与 FLOPs（十亿级）的表格，并可选写入 CSV。

### 常用命令示例

仅在 CPU 上跑 YOLO 系列：

```bash
python model_profiling.py --models "YOLO11-Small" "YOLOv8-Nano"
```

切换到 GPU（假设可用）并统一使用 512×512 输入：

```bash
python model_profiling.py --device cuda --img-size 512
```

保存结果到 CSV：

```bash
python model_profiling.py --output outputs/profile.csv
```

查看可选模型名称与全部参数：

```bash
python model_profiling.py --help
```

## 模型权重来源与说明

- **YOLO11/YOLOv8/RT-DETR**：通过 `ultralytics.YOLO(<weight>.pt)` 自动下载。当前 Ultralytics 官方提供 `rtdetr-l.pt`，脚本以其作为 RT-DETR-Nano 的近似基线，若有更轻量的官方权重可将 `ModelSpec.weight` 更新为实际文件名以替换。
- **Faster R-CNN**：使用 `torchvision` 自带的 `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` 预训练权重。
- **SSD-MobileNetV3-L**：使用 `SSDLite320_MobileNet_V3_Large_Weights.DEFAULT` 预训练权重。

> FLOPs 由 `thop` 返回的 MACs 直接换算为 FLOPs（单位：B），适合作为跨模型的相对对比指标。

## 输出格式示例

运行后会打印类似结果：

```
Model                     | Params (M) |  FLOPs (B)
----------------------------------------------------
YOLO11-Small              |      11.20 |      34.50
YOLO11-Nano               |       7.20 |      18.40
...
```

若指定 `--output path/to/file.csv`，文件内容为：

```csv
model,params_m,flops_b
YOLO11-Small,11.1950,34.5023
YOLO11-Nano,7.2164,18.3991
...
```
