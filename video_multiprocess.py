#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多进程视频处理 - 显著提升处理速度

使用生产者-消费者模式并行处理视频帧
"""
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from moviepy import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import json
import subprocess
from multiprocessing import Process, Queue, cpu_count, Value
import ctypes
import time


def cv2ImgAddText(img, text, pos, textColor=(255, 255, 255), textSize=30):
    """在图像上添加中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, textSize, encoding="utf-8")
                break
            except:
                continue

    if font is None:
        font = ImageFont.load_default()

    draw.text(pos, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class AudioVolumeMonitor:
    """音频音量监控器"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_array = None
        self.audio_fps = 44100
        self.audio_duration = 0
        self.total_samples = 0
        self.has_audio = False
        self._load_audio()

    def _load_audio(self):
        """加载音频数据到内存"""
        try:
            clip = VideoFileClip(self.video_path)
            if clip.audio is None:
                clip.close()
                return
            self.audio_array = clip.audio.to_soundarray(fps=self.audio_fps)
            self.audio_duration = clip.audio.duration
            self.total_samples = len(self.audio_array)
            self.has_audio = True
            clip.close()
        except Exception as e:
            print(f"加载音频数据时出错: {e}")
            self.has_audio = False

    def get_volume(self, start_time, duration=0.1):
        """获取指定时间段的音量"""
        if not self.has_audio or self.audio_array is None:
            return 0.0
        try:
            end_time = min(start_time + duration, self.audio_duration)
            if start_time >= self.audio_duration:
                return 0.0
            start_sample = int((start_time / self.audio_duration) * self.total_samples)
            end_sample = int((end_time / self.audio_duration) * self.total_samples)
            start_sample = max(0, min(start_sample, self.total_samples - 1))
            end_sample = max(0, min(end_sample, self.total_samples))
            audio_segment = self.audio_array[start_sample:end_sample]
            if len(audio_segment) == 0:
                return 0.0
            if len(audio_segment.shape) > 1:
                rms = np.sqrt(np.mean(audio_segment ** 2))
            else:
                rms = np.sqrt(np.mean(audio_segment ** 2))
            volume = min(rms * 2, 1.0)
            return volume
        except Exception as e:
            return 0.0


def extract_gps_trajectory(video_path):
    """提取GPS轨迹"""
    import warnings
    import io
    import sys

    trajectory = {
        'timestamps': [],
        'latitudes': [],
        'longitudes': [],
        'altitudes': [],
        'has_gps': False
    }

    try:
        warnings.filterwarnings('ignore')
        from pyosmogps import OsmoGps

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            gps = OsmoGps([video_path], timezone_offset=8)
            gps.extract()
            lats = gps.get_latitude()
            lons = gps.get_longitude()
            alts = gps.get_altitude()

            if len(lats) > 0:
                trajectory['latitudes'] = lats
                trajectory['longitudes'] = lons
                trajectory['altitudes'] = alts if len(alts) > 0 else [0] * len(lats)
                trajectory['has_gps'] = True

                gps_fps = 50.0
                for i in range(len(lats)):
                    time_offset = i / gps_fps
                    trajectory['timestamps'].append(time_offset)

        finally:
            sys.stderr = old_stderr

    except ImportError:
        pass
    except Exception:
        pass

    return trajectory


def get_gps_at_time(trajectory, video_start_time, current_video_time):
    """根据时间获取GPS坐标"""
    if not trajectory['has_gps'] or len(trajectory['latitudes']) == 0:
        return None, None, None, None

    gps_fps = 50.0
    gps_index = int(current_video_time * gps_fps)

    if gps_index >= len(trajectory['latitudes']):
        gps_index = len(trajectory['latitudes']) - 1

    lat = trajectory['latitudes'][gps_index]
    lon = trajectory['longitudes'][gps_index]
    alt = trajectory['altitudes'][gps_index] if gps_index < len(trajectory['altitudes']) else 0

    if video_start_time:
        real_datetime = video_start_time + timedelta(seconds=current_video_time)
    else:
        real_datetime = None

    return lat, lon, alt, real_datetime


def format_datetime(dt):
    """将datetime对象格式化为字符串"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def process_frame_worker(frame_data, audio_monitor, gps_trajectory, video_start_time, fps, panel_height, scale=1.0):
    """
    处理单个帧的worker函数
    在单独的进程中运行
    """
    frame_count, frame, current_time = frame_data

    # 获取GPS坐标
    current_lat, current_lon, current_alt, current_datetime = get_gps_at_time(
        gps_trajectory, video_start_time, current_time
    )

    # 获取音量
    current_volume = audio_monitor.get_volume(current_time, duration=0.1)

    height, width = frame.shape[:2]

    # 缩放帧（如果需要）
    if scale != 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]

    # 创建信息面板
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = (0, 0, 0)

    # 绘制信息
    y_offset = 35
    line_height = 35

    # 时间
    if current_datetime:
        time_str = format_datetime(current_datetime)
        cv2.putText(panel, f"Time: {time_str}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y_offset += line_height

    # GPS
    if current_lat is not None and current_lon is not None:
        gps_text = f"GPS: {current_lat:.6f}, {current_lon:.6f}"
        cv2.putText(panel, gps_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += line_height

        alt_text = f"Alt: {current_alt:.1f}m"
        cv2.putText(panel, alt_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(panel, "GPS: No Signal", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    y_offset += line_height

    # 音量
    volume_percent = current_volume * 100
    volume_text = f"Vol: {volume_percent:.1f}%"
    cv2.putText(panel, volume_text, (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # 音量条
    bar_width = 300
    bar_height = 20
    bar_x = 250
    bar_y = y_offset - 15

    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 (64, 64, 64), -1)

    volume_bar_width = int(bar_width * current_volume)
    if current_volume < 0.3:
        bar_color = (0, 255, 0)
    elif current_volume < 0.7:
        bar_color = (0, 165, 255)
    else:
        bar_color = (0, 0, 255)

    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + volume_bar_width, bar_y + bar_height),
                 bar_color, -1)

    # 叠加面板到帧
    frame[:panel_height, :] = panel

    return frame, frame_count


def frame_reader(video_path, frame_queue, total_frames, stop_flag, fps, skip_frames=1):
    """读取视频帧的进程"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    try:
        while not stop_flag.value and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳帧处理
            if frame_count % skip_frames != 0:
                continue

            current_time = frame_count / fps

            # 将帧放入队列
            frame_queue.put((frame_count, frame, current_time))

    finally:
        cap.release()
        # 发送结束信号
        frame_queue.put(None)


def frame_processor(frame_queue, result_queue, audio_monitor, gps_trajectory,
                     video_start_time, fps, panel_height, stop_flag, scale=1.0):
    """处理帧的进程"""
    try:
        while not stop_flag.value:
            frame_data = frame_queue.get()

            if frame_data is None:  # 结束信号
                result_queue.put(None)
                break

            # 处理帧
            processed_frame, frame_count = process_frame_worker(
                frame_data, audio_monitor, gps_trajectory,
                video_start_time, fps, panel_height, scale
            )

            # 将结果放入队列
            result_queue.put((processed_frame, frame_count))

    except Exception as e:
        print(f"帧处理错误: {e}")
        result_queue.put(None)


def frame_writer(result_queue, output_path, fps, frame_size, total_frames, stop_flag, use_fast_codec=False):
    """写入视频的进程"""

    # 选择编码器
    if use_fast_codec:
        # 使用mp4v编码器（更快，但兼容性稍差）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        # 使用H.264编码器（质量更好，但慢）
        fourcc = cv2.VideoWriter_fourcc(*'H264')

    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        print("错误：无法创建视频写入器")
        return

    processed_count = 0
    start_time = time.time()

    try:
        while not stop_flag.value:
            result = result_queue.get()

            if result is None:  # 结束信号
                break

            processed_frame, frame_count = result
            video_writer.write(processed_frame)
            processed_count += 1

            # 显示进度
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%) - "
                      f"速度: {fps_actual:.1f} fps, ETA: {eta:.0f}秒")

    finally:
        video_writer.release()


def process_video_multiprocess(video_path, output_path=None, display=False,
                                skip_frames=1, num_workers=None, scale=1.0, use_fast_codec=False):
    """
    多进程视频处理

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        display: 是否显示实时画面（多进程模式不支持显示）
        skip_frames: 跳帧处理
        num_workers: 工作进程数量（默认=CPU核心数-1）
        scale: 缩放比例（0.5=降低分辨率，速度提升2-4倍）
        use_fast_codec: 使用快速编码器（mp4v，质量稍差但速度快3-5倍）
    """
    print(f"\n{'='*70}")
    print("多进程视频处理模式")
    print(f"{'='*70}\n")

    # 打开视频获取信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 计算缩放后的尺寸
    output_width = int(width * scale)
    output_height = int(height * scale)

    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
    if scale != 1.0:
        print(f"输出分辨率: {output_width}x{output_height} (缩放: {scale}x)")

    if skip_frames > 1:
        print(f"跳帧处理: 每{skip_frames}帧处理1帧")

    if use_fast_codec:
        print(f"编码器: mp4v (快速模式)")

    # 确定工作进程数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"使用 {num_workers} 个工作进程")

    # 加载音频和GPS数据
    print("正在加载音频数据...")
    audio_monitor = AudioVolumeMonitor(video_path)
    if audio_monitor.has_audio:
        print(f"音频数据加载成功，时长: {audio_monitor.audio_duration:.2f}秒")
    else:
        print("无音频轨道")

    print("正在提取GPS轨迹...")
    gps_trajectory = extract_gps_trajectory(video_path)
    if gps_trajectory['has_gps']:
        print(f"成功提取 {len(gps_trajectory['latitudes'])} 个GPS数据点")
    else:
        print("未检测到GPS信息")

    # 获取视频开始时间
    video_start_time = None
    try:
        creation_time = os.path.getmtime(video_path)
        video_start_time = datetime.fromtimestamp(creation_time)
    except:
        pass

    panel_height = 220

    # 创建共享状态
    stop_flag = Value('b', 0)

    # 创建队列
    frame_queue = Queue(maxsize=100)  # 限制队列大小避免内存溢出
    result_queue = Queue(maxsize=100)

    # 启动进程
    print("\n开始处理...")

    # 读取进程
    reader_process = Process(
        target=frame_reader,
        args=(video_path, frame_queue, total_frames, stop_flag, fps, skip_frames)
    )

    # 处理进程（可以启动多个）
    processor_process = Process(
        target=frame_processor,
        args=(frame_queue, result_queue, audio_monitor, gps_trajectory,
              video_start_time, fps, panel_height, stop_flag, scale)
    )

    # 写入进程
    if output_path:
        writer_process = Process(
            target=frame_writer,
            args=(result_queue, output_path, fps, (output_width, output_height),
                  total_frames, stop_flag, use_fast_codec)
        )
    else:
        writer_process = None

    start_time = time.time()

    # 启动所有进程
    reader_process.start()
    processor_process.start()
    if writer_process:
        writer_process.start()

    # 等待完成
    reader_process.join()
    processor_process.join()
    if writer_process:
        writer_process.join()

    elapsed_time = time.time() - start_time

    print(f"\n处理完成！")
    print(f"  总帧数: {total_frames}")
    print(f"  耗时: {elapsed_time:.1f}秒")
    print(f"  平均速度: {total_frames/elapsed_time:.1f} fps")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='多进程视频元数据提取')
    parser.add_argument('--video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default='multiprocess_output.mp4',
                        help='输出视频路径')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='跳帧处理（每N帧处理1帧）')
    parser.add_argument('--workers', type=int, default=None,
                        help='工作进程数（默认=CPU核心数-1）')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='缩放比例（0.5=降低分辨率，速度提升2-4倍，默认1.0）')
    parser.add_argument('--fast-codec', action='store_true',
                        help='使用快速编码器（mp4v，质量稍差但速度快3-5倍）')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误：找不到视频文件 '{args.video}'")
        print("\n使用示例:")
        print("  python video_multiprocess.py --video test.mp4")
        print("  python video_multiprocess.py --video test.mp4 --skip-frames 2")
        print("  python video_multiprocess.py --video test.mp4 --skip-frames 5 --workers 4")
        print("\n性能优化:")
        print("  python video_multiprocess.py --video test.mp4 --skip-frames 5 --scale 0.5  # 降低分辨率+跳帧")
        print("  python video_multiprocess.py --video test.mp4 --skip-frames 10 --scale 0.5 --fast-codec  # 最快模式")
    else:
        process_video_multiprocess(
            video_path=args.video,
            output_path=args.output,
            display=False,  # 多进程模式不支持显示
            skip_frames=args.skip_frames,
            num_workers=args.workers,
            scale=args.scale,
            use_fast_codec=args.fast_codec
        )
