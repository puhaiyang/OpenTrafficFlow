#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合视频处理系统 - 车牌识别 + GPS + 音频 + 时间
结合YOLO车牌检测、LPRNet车牌识别、GPS轨迹提取和音量监控
"""
import cv2
import torch
import numpy as np
import os
from datetime import datetime, timedelta
from moviepy import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import json
import time
import io
import warnings

# 导入LPRNet模型
from model.LPRNet import build_lprnet

# ==================== 配置 ====================

# 车牌字符集
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
         'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         'I', 'O', '-']

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 模型加载 ====================

def load_models():
    """加载YOLO检测模型和LPRNet识别模型"""
    print("正在加载模型...")

    # 加载YOLO模型
    yolo_model = YOLO("weights/best.pt")

    # 加载LPRNet模型
    lpr_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    lpr_model.load_state_dict(torch.load("weights/Final_LPRNet_model.pth", map_location=device))
    lpr_model.to(device)
    lpr_model.eval()

    print("模型加载完成!")
    return yolo_model, lpr_model

# 全局模型加载
yolo_model = None
lpr_model = None

# ==================== 工具函数 ====================

def cv2ImgAddText(img, text, pos, textColor=(255, 255, 255), textSize=30):
    """在图像上添加中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
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

    draw.text(pos, text + " 侵走非机动车道", textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def decode_res(preds, chars):
    """CTC Greedy 解码"""
    if len(preds) == 0:
        return ""

    res = []
    blank_idx = len(chars) - 1

    for i in range(len(preds)):
        if preds[i] == blank_idx:
            continue
        if i > 0 and preds[i] == preds[i - 1]:
            continue
        res.append(chars[preds[i]])

    return "".join(res)


def format_datetime(dt):
    """将datetime对象格式化为字符串"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')


# ==================== 音频监控 ====================

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


# ==================== GPS轨迹提取 ====================

def extract_gps_trajectory(video_path):
    """提取GPS轨迹"""
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


# ==================== 车牌检测与识别 ====================

def detect_and_recognize_plates(frame, conf_threshold=0.5):
    """
    检测并识别车牌

    Args:
        frame: 输入图像帧
        conf_threshold: YOLO检测置信度阈值

    Returns:
        detections: 检测结果列表 [(plate_no, conf, bbox), ...]
    """
    global yolo_model, lpr_model

    if yolo_model is None or lpr_model is None:
        return []

    # YOLO 检测
    results = yolo_model(frame, conf=conf_threshold, verbose=False)
    detections = []

    for r in results:
        if len(r.boxes) == 0:
            continue

        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0])

            # 裁剪车牌
            crop_img = frame[y1:y2, x1:x2]

            # LPRNet 预处理
            tmp_img = cv2.resize(crop_img, (94, 24))
            tmp_img = tmp_img.astype('float32')
            tmp_img -= 127.5
            tmp_img *= 0.0078125
            tmp_img = np.transpose(tmp_img, (2, 0, 1))
            tmp_img = torch.from_numpy(tmp_img).unsqueeze(0).to(device)

            # LPRNet 推理
            with torch.no_grad():
                preds = lpr_model(tmp_img)
                preds = preds.cpu().numpy()
                arg_max_preds = np.argmax(preds, axis=1)
                plate_no = decode_res(arg_max_preds[0], CHARS)

            # 验证车牌长度（普通车牌7位，新能源车牌8位）
            if len(plate_no) in [7, 8]:
                detections.append((plate_no, conf, (x1, y1, x2, y2)))

    return detections


# ==================== 信息面板绘制 ====================

def draw_info_panel(frame, current_datetime, current_lat, current_lon, current_alt,
                    detected_plates, panel_height=180):
    """
    在视频帧上绘制信息面板

    Args:
        frame: 输入图像帧
        current_datetime: 当前时间
        current_lat, current_lon, current_alt: GPS坐标
        detected_plates: 检测到的车牌列表
        panel_height: 信息面板高度
    """
    height, width = frame.shape[:2]

    # 创建信息面板
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # 深灰色背景

    y_offset = 35
    line_height = 35
    x_pos = 20

    # 绘制分割线
    cv2.line(panel, (0, 0), (width, 0), (0, 200, 255), 3)

    # 时间
    if current_datetime:
        time_str = format_datetime(current_datetime)
        cv2.putText(panel, f"时间: {time_str}", (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y_offset += line_height

    # GPS
    if current_lat is not None and current_lon is not None:
        gps_text = f"GPS: {current_lat:.6f}, {current_lon:.6f}"
        cv2.putText(panel, gps_text, (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += line_height

        alt_text = f"海拔: {current_alt:.1f}m"
        cv2.putText(panel, alt_text, (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(panel, "GPS: 无信号", (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    y_offset += line_height

    # 车牌信息
    if detected_plates:
        cv2.putText(panel, f"检测到车牌: {len(detected_plates)}", (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 叠加面板到帧（放在底部）
    frame[-panel_height:, :] = panel

    return frame


def draw_plate_boxes(frame, detections):
    """在帧上绘制车牌检测框"""
    for plate_no, conf, bbox in detections:
        x1, y1, x2, y2 = bbox
        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 添加车牌文字
        frame = cv2ImgAddText(frame, plate_no, (x1 - 100, y1 - 100),
                              textColor=(255, 0, 0), textSize=60)
    return frame


# ==================== 主处理函数 ====================

def process_video(video_path, output_path=None, display=True,
                 save_output=False, conf_threshold=0.5, skip_frames=1):
    """
    处理视频文件，进行车牌检测、GPS轨迹、音量监控和时间显示

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        display: 是否显示实时画面
        save_output: 是否保存输出视频
        conf_threshold: 检测置信度阈值
        skip_frames: 跳帧处理（每N帧处理1帧）
    """
    global yolo_model, lpr_model

    # 加载模型
    if yolo_model is None or lpr_model is None:
        yolo_model, lpr_model = load_models()

    print(f"\n{'='*70}")
    print("综合视频处理系统 - 车牌识别 + GPS + 音频 + 时间")
    print(f"{'='*70}\n")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

    # 加载音频数据
    print("正在加载音频数据...")
    audio_monitor = AudioVolumeMonitor(video_path)
    if audio_monitor.has_audio:
        print(f"音频数据加载成功，时长: {audio_monitor.audio_duration:.2f}秒")
    else:
        print("无音频轨道")

    # 提取GPS轨迹
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
        print(f"视频创建时间: {format_datetime(video_start_time)}")
    except:
        pass

    # 创建视频写入器
    video_writer = None
    if save_output and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"输出视频: {output_path}")

    frame_count = 0
    processed_count = 0
    all_detections = []

    panel_height = 250
    start_time = time.time()

    print("\n开始处理... 按 'q' 退出，按 ' ' 暂停")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 跳帧处理
        if frame_count % skip_frames != 0:
            # 如果需要写入视频，仍然写入原帧
            if video_writer:
                video_writer.write(frame)
            continue

        processed_count += 1

        # 计算当前时间
        current_time = frame_count / fps

        # 获取GPS坐标
        current_lat, current_lon, current_alt, current_datetime = get_gps_at_time(
            gps_trajectory, video_start_time, current_time
        )

        # 车牌检测与识别
        detections = detect_and_recognize_plates(frame, conf_threshold)

        # 保存检测结果
        for plate_no, conf, bbox in detections:
            all_detections.append({
                'frame': frame_count,
                'plate': plate_no,
                'confidence': conf,
                'bbox': bbox,
                'time': current_time
            })

        # 绘制车牌检测框
        frame = draw_plate_boxes(frame, detections)

        # 绘制信息面板
        frame = draw_info_panel(frame, current_datetime, current_lat, current_lon,
                               current_alt, detections, panel_height)

        # 写入输出视频
        if video_writer:
            video_writer.write(frame)

        # 显示进度
        if processed_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = processed_count / elapsed if elapsed > 0 else 0
            print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%) - "
                  f"速度: {fps_actual:.1f} fps")

        # 显示画面
        if display:
            # 添加帧数信息
            info_text = f"Frame: {frame_count}/{total_frames} | Detected: {len(detections)}"
            cv2.putText(frame, info_text, (10, height - panel_height + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("综合视频处理系统", frame)

            # 按 'q' 退出，按 ' ' 暂停
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n处理被用户终止")
                break
            elif key == ord(' '):
                print("已暂停 - 按任意键继续...")
                cv2.waitKey(0)

    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("处理完成!")
    print(f"{'='*70}")
    print(f"总帧数: {total_frames}")
    print(f"处理帧数: {processed_count}")
    print(f"耗时: {elapsed_time:.1f}秒")
    print(f"平均速度: {processed_count/elapsed_time:.1f} fps")
    print(f"总检测数: {len(all_detections)}")

    # 输出检测统计
    if all_detections:
        print("\n--- 车牌检测统计 ---")
        plate_counts = {}
        for det in all_detections:
            plate = det['plate']
            plate_counts[plate] = plate_counts.get(plate, 0) + 1

        print("检测最多的车牌 (Top 10):")
        for plate, count in sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {plate}: {count} 次")

    return all_detections


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='综合视频处理系统 - 车牌识别 + GPS + 音频 + 时间')
    parser.add_argument('--video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出视频路径（可选）')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示实时画面')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='检测置信度阈值（默认: 0.5）')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='跳帧处理，每N帧处理1帧（默认: 1，不跳帧）')

    args = parser.parse_args()

    # 检查输入视频是否存在
    if not os.path.exists(args.video):
        print(f"错误：找不到视频文件 '{args.video}'")
        print("\n使用示例:")
        print("  python video_plate_with_metadata.py --video test.mp4")
        print("  python video_plate_with_metadata.py --video test.mp4 --output result.mp4")
        print("  python video_plate_with_metadata.py --video test.mp4 --no-display --conf 0.6")
        print("  python video_plate_with_metadata.py --video test.mp4 --skip-frames 5  # 跳帧加速")
    else:
        process_video(
            video_path=args.video,
            output_path=args.output,
            display=not args.no_display,
            save_output=bool(args.output),
            conf_threshold=args.conf,
            skip_frames=args.skip_frames
        )
