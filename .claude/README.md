# OpenTrafficFlow - Claude Code 项目说明

## 环境要求

**重要**: 此项目必须在 conda 环境 `ultralytics-env` 中运行。

### 运行命令的方式

所有 Python 相关的命令都需要在 `ultralytics-env` conda 环境中执行：

```bash
# Windows 激活环境方式
conda activate ultralytics-env

# 或使用 conda run 直接执行
conda run -n ultralytics-env <command>
```

### Python 命令示例

```bash
# 安装依赖
conda run -n ultralytics-env pip install -r requirements.txt

# 运行脚本
conda run -n ultralytics-env python video_metadata_display.py --video test.mp4

# 检查包
conda run -n ultralytics-env pip list
```

### 可用脚本

- `video_plate_detection.py` - 车牌检测与识别
- `video_metadata_display.py` - 视频元数据提取（时间、GPS、音量）
  - **支持实时GPS轨迹显示**（每一帧显示对应的GPS坐标）
  - **显示真实GPS时间**而不是播放时长
  - 显示海拔信息
  - **高效的音频音量监控**（使用AudioVolumeMonitor类，一次性加载音频数据）
- `yolo_test.py` - YOLO 测试脚本
- `yolo_final_test.py` - YOLO 最终测试脚本
- `test_pyosmogps.py` - PyOsmoGPS GPS 提取测试脚本
- `test_metadata.py` - 视频元数据提取测试脚本
- `test_gps_trajectory.py` - GPS轨迹提取测试脚本
- `test_audio_volume.py` - 音频音量监控测试脚本

### 测试视频

**默认测试视频路径**: `F:\video\自动违章举报\DJI_20251223081916_0084_D.mp4`

这是 DJI Osmo Action 4 录制的视频，包含 GPS 元数据，可用于测试：
- 视频元数据提取功能（时间、GPS、音量）
- PyOsmoGPS 库的 GPS 提取功能

**快速测试命令**:
```bash
# 测试视频元数据提取（包括 GPS）
python video_metadata_display.py --video F:\video\自动违章举报\DJI_20251223081916_0084_D.mp4
```

### 依赖库

主要依赖包括：
- torch, torchvision
- opencv-python
- ultralytics (YOLO)
- moviepy (视频处理)
- pyosmogps (DJI 视频 GPS 提取)

### GPU 支持

项目支持 NVIDIA GPU 加速（通过 CUDA），确保在正确的 conda 环境中安装了 PyTorch CUDA 版本。
