# CCPD 数据集转换为 YOLO 格式 - 使用指南

## 概述

`convert_all_ccpd_to_yolo.py` 支持将三种 CCPD 数据集转换为 YOLO 格式：
- **CCPD-2019**: 标准格式，包含多个子目录（ccpd_base, ccpd_blur, 等）
- **CCPD-2020**: 标准格式，包含 train/val/test 划分
- **CCPD-BlueGreenYellow**: 特殊格式，文件名带 `.jpg.png` 扩展名

## 数据集格式说明

### CCPD-2019 和 CCPD-2020

文件名格式：`025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg`

- `025`: 车牌倾斜角度
- `95_113`: 边界框左上角坐标 (x0, y0)
- `154&383`: 边界框右下角坐标 (x1, y1)
- 后续部分包含四个顶点坐标、车牌类型等信息

### CCPD-BlueGreenYellow

文件名格式：`0-0_0-0&342_719&610-714&610_0&585_15&342_719&367-29_16_2_2_30_31_28-0-0.jpg.png`

- 该数据集使用更复杂的坐标编码
- 脚本会自动检测并提取边界框信息
- 支持双重扩展名 `.jpg.png`

## 使用方法

### 1. 转换单个数据集

```bash
# 转换 CCPD-2019
python convert_all_ccpd_to_yolo.py \
    --source ./CCPD_Datasets/CCPD/puhaiyang___CCPD2019/CCPD2019 \
    --target ./YOLO_Data/CCPD2019 \
    --dataset-type ccpd2019 \
    --copy

# 转换 CCPD-2020（保留原始 train/val/test 划分）
python convert_all_ccpd_to_yolo.py \
    --source ./CCPD_Datasets/CCPD2020/puhaiyang___CCPD2020/CCPD2020 \
    --target ./YOLO_Data/CCPD2020 \
    --dataset-type ccpd2020 \
    --preserve-splits \
    --copy

# 转换 CCPD-BlueGreenYellow
python convert_all_ccpd_to_yolo.py \
    --source ./CCPD_Datasets/CCPD_BlueGreenYellow/puhaiyang___ccpdblueyellowgreen/ccpd_blue_yellow_green \
    --target ./YOLO_Data/BlueGreenYellow \
    --dataset-type bluegreenyellow \
    --copy
```

### 2. 转换所有数据集（一键转换）

```bash
python convert_all_ccpd_to_yolo.py \
    --source ./CCPD_Datasets \
    --target ./YOLO_Data \
    --all \
    --copy
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 源数据集路径 | `./CCPD_Datasets` |
| `--target` | 目标路径 | `./YOLO_Data` |
| `--dataset-type` | 数据集类型 | `auto` |
| `--val-ratio` | 验证集比例 | `0.2` |
| `--test-ratio` | 测试集比例 | `0.1` |
| `--copy` | 复制图片（保留原文件） | `True` |
| `--no-yaml` | 不创建 data.yaml | `False` |
| `--preserve-splits` | 保留原始划分 | `False` |
| `--all` | 转换所有数据集 | `False` |

## 输出目录结构

转换后的 YOLO 数据集结构：

```
YOLO_Data/
├── CCPD2019/                # CCPD-2019 数据集
│   ├── data.yaml            # YOLO 配置文件
│   ├── images/
│   │   ├── train/           # 训练集图片
│   │   ├── val/             # 验证集图片
│   │   └── test/            # 测试集图片
│   └── labels/
│       ├── train/           # 训练集标签
│       ├── val/             # 验证集标签
│       └── test/            # 测试集标签
├── CCPD2020/                # CCPD-2020 数据集
│   └── ...
└── CCPD-BlueGreenYellow/    # CCPD-BlueGreenYellow 数据集
    └── ...
```

## 标签格式

YOLO 格式标签文件（每个图片对应一个 `.txt` 文件）：

```
0 x_center y_center width height
```

例如：
```
0 0.523456 0.623456 0.123456 0.056789
```

所有值已归一化到 [0, 1] 范围。

## 训练示例

### YOLOv5

```bash
python train.py \
    --data ./YOLO_Data/CCPD2020/data.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16
```

### YOLOv8

```bash
yolo detect train \
    data=./YOLO_Data/CCPD2020/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    batch=16
```

## 常见问题

### Q: 转换失败，提示 "无法解析文件名"

**A**: 检查以下几点：
1. 确认 `--dataset-type` 参数正确
2. 对于 CCPD-BlueGreenYellow，确认文件名格式符合规范
3. 查看具体文件名是否与示例格式一致

### Q: CCPD-BlueGreenYellow 解析失败

**A**: 该数据集格式较为特殊，脚本支持以下格式：
- 标准格式（部分文件）
- 扩展格式（需要从多个坐标点计算边界框）

如果仍有问题，可以查看脚本中的 `extract_bbox_ccpd_blueyellow` 函数进行调试。

### Q: 内存不足

**A**: 对于大型数据集（如 CCPD-2019），建议：
1. 使用 `--copy` 参数（默认）
2. 分批转换各个子目录
3. 增加系统交换空间

### Q: 想要合并多个数据集

**A**: 可以分别转换后手动合并：
```bash
# 分别转换
python convert_all_ccpd_to_yolo.py --all --target ./YOLO_Data

# 手动合并（可选）
# 将 images/train, images/val, labels/train, labels/val 合并
```

## 测试转换结果

使用提供的验证脚本：

```bash
python verify_yolo_dataset.py --dataset ./YOLO_Data/CCPD2020
```

这将检查：
1. 图片和标签文件数量是否匹配
2. 标签格式是否正确
3. 坐标是否在有效范围内

## 支持的数据集路径

脚本会自动检测以下路径结构：

```
CCPD_Datasets/
├── CCPD/
│   └── puhaiyang___CCPD2019/
│       └── CCPD2019/
│           ├── ccpd_base/
│           ├── ccpd_blur/
│           └── ...
├── CCPD2020/
│   └── puhaiyang___CCPD2020/
│       └── CCPD2020/
│           └── ccpd_green/
│               ├── train/
│               ├── val/
│               └── test/
└── CCPD_BlueGreenYellow/
    └── puhaiyang___ccpdblueyellowgreen/
        └── ccpd_blue_yellow_green/
            └── *.jpg.png
```

如果您的目录结构不同，请使用 `--source` 参数指定正确的路径。
