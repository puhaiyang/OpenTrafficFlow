# CCPD 数据集转换为 YOLO 格式 - 使用指南

## 功能说明

`convert_ccpd_to_yolo.py` 脚本用于将 CCPD 数据集转换为 YOLOv5 训练所需的格式。

### 主要功能

1. **自动解析 CCPD 文件名** - 从文件名中提取车牌边界框坐标
2. **坐标格式转换** - CCPD 格式 → YOLO 格式（归一化坐标）
3. **数据集划分** - 自动划分训练集/验证集/测试集
4. **目录结构创建** - 自动创建 YOLOv5 标准目录结构
5. **配置文件生成** - 自动生成 data.yaml 配置文件

---

## CCPD vs YOLO 格式对比

### CCPD 格式

文件名包含坐标信息：
```
025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg
```

边界框格式：
```
x0, y0, x1, y1  (左上角和右下角坐标)
```

### YOLO 格式

标签文件 (`.txt`)：
```
0 0.523456 0.456789 0.123456 0.098765
```

格式：`class_id x_center y_center width height` (归一化到 0-1)

---

## 使用方法

### 基础用法

```bash
# 使用默认路径
python convert_ccpd_to_yolo.py

# 指定 CCPD 数据集路径
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020

# 指定输出路径
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020 --target ./YOLO_Dataset
```

### 高级用法

```bash
# 复制图片（保留原始文件）
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020 --copy

# 自定义数据集划分比例
python convert_ccpd_to_yolo.py --val-ratio 0.15 --test-ratio 0.05

# 不生成配置文件
python convert_ccpd_to_yolo.py --no-yaml

# 完整示例
python convert_ccpd_to_yolo.py \
    --source ./CCPD_Datasets/CCPD2020 \
    --target ./my_dataset \
    --val-ratio 0.2 \
    --test-ratio 0.1 \
    --copy
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--source` | str | `./CCPD_Datasets/CCPD2020` | CCPD 数据集源路径 |
| `--target` | str | `./YOLO_Data` | YOLO 数据集输出路径 |
| `--val-ratio` | float | `0.2` | 验证集比例 (0-1) |
| `--test-ratio` | float | `0.1` | 测试集比例 (0-1) |
| `--copy` | flag | `False` | 复制图片（默认移动） |
| `--no-yaml` | flag | `False` | 不生成 data.yaml |

---

## 目录结构

### 输入（CCPD 数据集）

```
CCPD_Datasets/CCPD2020/
├── train/
│   ├── 001-1_1-123&234_345&456-0&0_0&0_0&0-0-0.jpg
│   ├── 002-2_2-234&345_456&567-0&0_0&0_0&0-0-0.jpg
│   └── ...
└── test/
    ├── ...
```

### 输出（YOLO 数据集）

```
YOLO_Data/
├── images/
│   ├── train/
│   │   ├── ccpd_train_000001.jpg
│   │   ├── ccpd_train_000002.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── ccpd_train_000001.txt  # 标签文件
│   │   ├── ccpd_train_000002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── data.yaml  # 数据集配置文件
```

### 标签文件示例

`ccpd_train_000001.txt`:
```
0 0.523456 0.456789 0.123456 0.098765
```

格式：`class_id x_center y_center width height`

---

## 完整工作流程

### 步骤 1: 下载 CCPD 数据集

```bash
python download_ccpd_windows.py
# 选择选项 1 - 下载 CCPD2020 (865MB)
```

### 步骤 2: 解压数据集

脚本会自动解压，或手动解压：
```bash
# Windows
cd CCPD_Datasets\CCPD2020
tar -xf CCPD2020.zip

# Linux/Mac
unzip CCPD_Datasets/CCPD2020/CCPD2020.zip -d CCPD_Datasets/CCPD2020/
```

### 步骤 3: 转换为 YOLO 格式

```bash
python convert_ccpd_to_yolo.py \
    --source ./CCPD_Datasets/CCPD2020 \
    --target ./YOLO_Data \
    --copy
```

### 步骤 4: 训练 YOLOv5

```bash
# 克隆 YOLOv5 仓库（如果还没有）
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 安装依赖
pip install -r requirements.txt

# 开始训练
python train.py \
    --data ../YOLO_Data/data.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16
```

---

## 常见问题

### Q1: 脚本报错 "找不到图片文件"

**A**: 检查 `--source` 路径是否正确，应该是包含 `train/test` 文件夹的父目录。

```bash
# 正确
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020

# 错误
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020/train
```

### Q2: 解析文件名失败

**A**: CCPD 文件名格式必须正确：
```
正确: 025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg
错误: image_001.jpg
```

### Q3: 数据集划分比例

**A**: 默认比例为：
- 训练集: 70% (val_ratio + test_ratio 的剩余部分)
- 验证集: 20%
- 测试集: 10%

可以自定义：
```bash
python convert_ccpd_to_yolo.py --val-ratio 0.15 --test-ratio 0.05
```

### Q4: 内存不足

**A**: 如果数据集很大，可以：
1. 只处理部分数据（移动部分图片到临时文件夹）
2. 使用 `--copy` 选项避免重复处理

### Q5: 验证转换结果

**A**: 检查标签文件内容：
```bash
# 查看标签文件
cat YOLO_Data/labels/train/ccpd_train_000001.txt

# 应该输出类似：
# 0 0.523456 0.456789 0.123456 0.098765
```

---

## 技术细节

### 坐标转换公式

```
CCPD: (x0, y0, x1, y1)
  ↓
YOLO: (x_center, y_center, width, height)

x_center = (x0 + x1) / 2 / image_width
y_center = (y0 + y1) / 2 / image_height
width = (x1 - x0) / image_width
height = (y1 - y0) / image_height
```

### CCPD 文件名解析

```
025-95_113-154&383_386&473-0&0_0&0_0&0-0-0.jpg
  │   │     │     │
  │   │     │     └─ 右下角坐标: (386, 473)
  │   │     └─ 左上角坐标: (95, 113)
  │   └─ 分隔符
  └─ 车牌倾斜角度（不用于检测任务）
```

正则表达式：
```python
r'-(\d+)_(\d+)-(\d+)&(\d+)_'
```

---

## 示例输出

```
======================================================================
CCPD 数据集转换为 YOLO 格式
======================================================================
✓ 已创建 YOLO 数据集目录结构: ./YOLO_Data

正在扫描图片文件...
✓ 找到 10000 个图片文件

正在划分数据集...
✓ 训练集: 7000 张
✓ 验证集: 2000 张
✓ 测试集: 1000 张

======================================================================
处理 train 数据集...
======================================================================
处理 train: 100%|██████████| 7000/7000 [02:15<00:00, 51.23it/s]

train 数据集处理完成:
  成功: 6985
  失败: 15

[...]

======================================================================
转换完成!
======================================================================
总计成功: 9990 张
总计失败: 10 张

YOLO 数据集已保存到: ./YOLO_Data
✓ 已创建数据集配置文件: ./YOLO_Data/data.yaml
```

---

## 相关文件

- `download_ccpd.py` - 数据集下载脚本
- `download_ccpd_windows.py` - Windows 优化版下载脚本
- `MANUAL_DOWNLOAD_GUIDE.md` - 手动下载指南
- `convert_ccpd_to_yolo.py` - 格式转换脚本（本文件）
