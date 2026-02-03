# 📊 数据集说明

本项目使用的数据集文件未包含在 Git 仓库中，需要单独下载。

## 数据集列表

### 1. CCPD2020 数据集（推荐）

- **大小**: 865.7 MB
- **用途**: 车牌检测训练
- **格式**: CCPD → 自动转换为 YOLO 格式
- **下载方式**: 运行自动下载脚本

### 2. CCPD 数据集（可选）

- **大小**: 12.6 GB
- **用途**: 更大规模的车牌检测训练
- **格式**: CCPD → 自动转换为 YOLO 格式
- **下载方式**: 运行自动下载脚本或使用专业下载工具

## 快速开始

### Windows 用户

```batch
# 方式 1: 使用快速启动脚本
quick_start.bat

# 方式 2: 手动执行
python download_ccpd_windows.py
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020 --target ./YOLO_Data --copy
```

### Linux/Mac 用户

```bash
# 设置环境变量
export OPENXLAB_AK=your_access_key
export OPENXLAB_SK=your_secret_key

# 下载并转换
python download_ccpd.py
python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020 --target ./YOLO_Data --copy
```

## 目录结构

下载和转换后的目录结构：

```
OpenTrafficFlow/
├── CCPD_Datasets/          # 原始 CCPD 数据集（Git 忽略）
│   ├── CCPD2020/
│   │   ├── CCPD2020.zip
│   │   └── ... (解压后)
│   └── CCPD/
│       └── ...
│
├── YOLO_Data/              # YOLO 格式数据集（Git 忽略）
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
│
└── ... (其他代码文件)
```

## 详细说明

- **下载脚本**: [download_ccpd.py](download_ccpd.py) / [download_ccpd_windows.py](download_ccpd_windows.py)
- **转换脚本**: [convert_ccpd_to_yolo.py](convert_ccpd_to_yolo.py)
- **验证脚本**: [verify_yolo_dataset.py](verify_yolo_dataset.py)
- **下载指南**: [MANUAL_DOWNLOAD_GUIDE.md](MANUAL_DOWNLOAD_GUIDE.md)
- **转换指南**: [CCPD_TO_YOLO_GUIDE.md](CCPD_TO_YOLO_GUIDE.md)

## .gitignore 配置

数据集文件夹已在 `.gitignore` 中配置：

```gitignore
# CCPD 原始数据集
CCPD_Datasets/

# YOLO 格式数据集
YOLO_Data/

# 数据集压缩文件
*.zip
*.tar.gz
*.tar.xz
```

## 重新下载

如果数据集丢失或损坏：

```bash
# 删除现有数据集
rm -rf CCPPD_Datasets YOLO_Data  # Linux/Mac
rmdir /s /q CCPPD_Datasets YOLO_Data  # Windows

# 重新下载
python download_ccpd_windows.py
```

## 获取 OpenXLab 凭证

1. 访问 https://openxlab.org.cn
2. 注册账号
3. 获取 Access Key 和 Secret Key
4. 运行脚本时输入凭证或设置环境变量

## 常见问题

### Q: 为什么数据集不在 Git 仓库中？

A: 数据集文件很大（GB 级别），不适合提交到 Git。Git 仓库只包含代码和文档。

### Q: 克隆项目后如何获取数据集？

A: 运行 `python download_ccpd_windows.py` 或查看详细指南。

### Q: 下载数据集需要多长时间？

A: 取决于网络速度：
- CCPD2020 (865MB): 约 5-15 分钟
- CCPD (12.6GB): 约 1-3 小时

### Q: 下载中断了怎么办？

A: 重新运行下载脚本，支持断点续传。

### Q: 可以使用其他数据集吗？

A: 可以，但需要转换为 YOLO 格式。参考 [convert_ccpd_to_yolo.py](convert_ccpd_to_yolo.py) 修改转换逻辑。
