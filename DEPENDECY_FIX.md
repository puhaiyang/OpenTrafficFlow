# scipy/sklearn DLL 错误修复说明

## 问题描述

```
ImportError: DLL load failed while importing _propack: 找不到指定的模块。
```

这是 Windows 系统下常见的 scipy/scikit-learn 依赖问题。

## 已修复

✅ 已修改 [convert_ccpd_to_yolo.py](convert_ccpd_to_yolo.py)，移除 sklearn 依赖
✅ 改用 Python 内置的 `random` 模块实现数据集划分
✅ 无需安装额外的机器学习库

## 现在可以运行

```bash
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020 --target ./YOLO_Data --copy
```

## 修改内容

### 之前（使用 sklearn）
```python
from sklearn.model_selection import train_test_split

train_files, temp_files = train_test_split(
    image_files,
    test_size=(val_ratio + test_ratio),
    random_state=42
)
```

### 现在（使用内置 random）
```python
import random

random.seed(42)
shuffled_files = image_files.copy()
random.shuffle(shuffled_files)

# 手动划分
train_files = shuffled_files[:train_count]
val_files = shuffled_files[train_count:train_count + val_count]
test_files = shuffled_files[train_count + val_count:]
```

## 如果仍然需要 sklearn 的其他功能

可以尝试以下修复方法：

### 方法 1: 重新安装 scipy

```bash
pip uninstall scipy scikit-learn -y
pip install scipy scikit-learn
```

### 方法 2: 安装 Microsoft Visual C++ Redistributable

下载并安装：https://aka.ms/vs/17/release/vc_redist.x64.exe

### 方法 3: 使用 conda（如果使用 conda 环境）

```bash
conda install -c anaconda scipy scikit-learn
```

## 推荐方案

**使用已修复的脚本**，无需安装 sklearn，更加轻量。
