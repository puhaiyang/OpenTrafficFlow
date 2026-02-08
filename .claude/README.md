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
- `video_multiprocess.py` - **多进程视频处理**（最快速度）
  - 使用生产者-消费者模式并行处理视频帧
  - 三个独立进程：读取进程、处理进程、写入进程
  - 避免Python GIL限制，充分利用多核CPU
  - **推荐用于大批量视频处理**
- `yolo_test.py` - YOLO 测试脚本
- `yolo_final_test.py` - YOLO 最终测试脚本
- `test_pyosmogps.py` - PyOsmoGPS GPS 提取测试脚本
- `test_metadata.py` - 视频元数据提取测试脚本
- `test_gps_trajectory.py` - GPS轨迹提取测试脚本
- `test_audio_volume.py` - 音频音量监控测试脚本
- `performance_guide.py` - 性能优化指南和性能对比分析

### 测试视频

**默认测试视频路径**: `F:\video\自动违章举报\DJI_20251223081916_0084_D.mp4`

这是 DJI Osmo Action 4 录制的视频，包含 GPS 元数据，可用于测试：
- 视频元数据提取功能（时间、GPS、音量）
- PyOsmoGPS 库的 GPS 提取功能

**快速测试命令**:
```bash
# 标准模式（高质量）
python video_metadata_display.py --video F:\video\自动违章举报\DJI_20251223081916_0084_D.mp4

# 性能优化模式（推荐，5倍速）
python video_metadata_display.py --video F:\video\自动违章举报\DJI_20251223081916_0084_D.mp4 --skip-frames 5 --no-display

# 快速预览（10倍速）
python video_metadata_display.py --video F:\video\自动违章举报\DJI_20251223081916_0084_D.mp4 --skip-frames 10 --no-display

# 多进程模式（最快，适合大批量处理）
python video_multiprocess.py --video F:\video\自动违章举报\DJI_20251223081916_0084_D.mp4 --output multiprocess_output.mp4 --skip-frames 5
```

### 性能优化建议

**video_metadata_display.py 性能优化参数**：

| 参数 | 说明 | 效果 |
|------|------|------|
| `--skip-frames N` | 跳帧处理（每N帧处理1帧） | 速度提升N倍 |
| `--no-display` | 不显示实时预览窗口 | 额外提升20-30% |
| `--slow-codec` | 使用慢速编码器（默认使用快速H.264） | - |

**推荐配置**：
- **快速预览**: `--skip-frames 5 --no-display` （5倍速）
- **高质量输出**: `--skip-frames 2` （2倍速，画质基本无损）
- **最快速度**: `--skip-frames 10 --no-display --no-save` （10倍速，不保存输出）

**性能对比**（6分钟视频，18,710帧）：
- 原始模式：~10分钟
- 跳帧5倍：~2分钟
- 跳帧10倍：~1分钟

### 多进程处理模式（最快）

**video_multiprocess.py** - 使用多进程并行处理视频：

**优势**：
- 使用生产者-消费者模式，三个独立进程并行工作
- 避免Python GIL限制，充分利用多核CPU
- 适合大批量视频处理

**使用方法**：
```bash
# 基本用法
python video_multiprocess.py --video input.mp4 --output output.mp4

# 跳帧处理（5倍速）
python video_multiprocess.py --video input.mp4 --output output.mp4 --skip-frames 5

# 指定工作进程数
python video_multiprocess.py --video input.mp4 --output output.mp4 --workers 8
```

**性能对比**：
| 模式 | 6分钟视频处理时间 | 加速比 |
|------|------------------|--------|
| 单进程原始模式 | ~10分钟 | 1x |
| 单进程跳帧5倍 | ~2分钟 | 5x |
| **多进程跳帧5倍** | **~1分钟** | **10x** |
| **多进程跳帧10倍** | **~30秒** | **20x** |

查看完整性能优化指南：`python performance_guide.py`

### 依赖库

主要依赖包括：
- torch, torchvision
- opencv-python
- ultralytics (YOLO)
- moviepy (视频处理)
- pyosmogps (DJI 视频 GPS 提取)

### GPU 支持

项目支持 NVIDIA GPU 加速（通过 CUDA），确保在正确的 conda 环境中安装了 PyTorch CUDA 版本。
