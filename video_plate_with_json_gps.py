#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合视频处理系统 - 车牌识别 + GPS（从JSON）+ 时间

与 video_plate_with_metadata.py 功能相同，但 GPS 数据从 JSON 文件读取
适用于 trim_video.py 生成的视频和 GPS JSON 文件
"""
import cv2
import torch
import numpy as np
import os
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import json
import time
import warnings
import threading
import queue

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

# ==================== 字体加载（全局只加载一次）====================

_font_cache = {}  # 字体缓存: {textSize: font_object}

def _get_font(textSize=30):
    """获取字体对象（使用缓存，只加载一次）"""
    global _font_cache

    # 如果缓存中已有，直接返回
    if textSize in _font_cache:
        return _font_cache[textSize]

    # 字体路径列表
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
                print(f"[INFO] 已加载字体: {font_path} (size={textSize})")
                break
            except:
                continue

    if font is None:
        font = ImageFont.load_default()
        print(f"[WARNING] 使用默认字体")

    # 缓存字体
    _font_cache[textSize] = font
    return font

# ==================== 工具函数 ====================

def cv2ImgAddText(img, text, pos, textColor=(255, 255, 255), textSize=30):
    """在图像上添加中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # 使用缓存的字体
    font = _get_font(textSize)

    draw.text(pos, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def format_datetime(dt):
    """将datetime对象格式化为字符串"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')


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


# ==================== GPS从JSON读取 ====================

def load_gps_from_json(json_path):
    """
    从JSON文件加载GPS数据

    Args:
        json_path: GPS JSON文件路径

    Returns:
        gps_data: 包含GPS数据的字典，或None
    """
    if not os.path.exists(json_path):
        print(f"[WARNING] GPS JSON 文件不存在: {json_path}")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"[INFO] 成功加载 GPS JSON: {data.get('total_points', 0)} 个数据点")
        return data
    except Exception as e:
        print(f"[ERROR] 读取 GPS JSON 失败: {e}")
        return None


def get_gps_at_time_from_json(gps_data, video_time):
    """
    从GPS JSON数据中获取指定时间的GPS坐标

    Args:
        gps_data: GPS数据字典
        video_time: 视频中的时间（秒）

    Returns:
        lat, lon, alt: GPS坐标，如果没有则返回None
    """
    if gps_data is None:
        return None, None, None

    data_points = gps_data.get('data', [])
    if not data_points:
        return None, None, None

    gps_fps = gps_data.get('gps_fps', 50.0)

    # 根据时间计算索引
    index = int(video_time * gps_fps)

    if index >= len(data_points):
        index = len(data_points) - 1

    point = data_points[index]
    return point.get('latitude'), point.get('longitude'), point.get('altitude')


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
    """在视频帧上绘制信息面板（已弃用，保留用于兼容）"""
    return frame


def draw_plate_boxes(frame, detections, current_datetime=None, current_lat=None, current_lon=None, current_alt=None):
    """
    在帧上绘制车牌检测框，并显示时间和GPS信息

    Args:
        frame: 输入图像帧
        detections: 检测到的车牌列表
        current_datetime: 当前时间
        current_lat, current_lon: GPS坐标
    """
    for plate_no, conf, bbox in detections:
        x1, y1, x2, y2 = bbox

        # 计算文本位置
        text_x = x1 - 100
        if text_x < 0:
            text_x = 10
        text_y_start = y1 - 250
        if text_y_start < 0:
            text_y_start = 10

        line_height = 50
        current_y = text_y_start

        # 绘制GPS坐标和时间信息（在车牌上方）
        if current_lat is not None and current_lon is not None and current_alt is not None:
            gps_text = f"GPS经纬度: {current_lat:.6f}, {current_lon:.6f} 海拔:{current_alt:.1f}m"
            frame = cv2ImgAddText(frame, gps_text, (text_x, current_y),
                                  textColor=(0, 255, 0), textSize=40)
            current_y += line_height

        if current_datetime:
            time_str = format_datetime(current_datetime)
            frame = cv2ImgAddText(frame, f"时间: {time_str}", (text_x, current_y),
                                  textColor=(0, 255, 255), textSize=40)
            current_y += line_height

        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 添加车牌文字
        frame = cv2ImgAddText(frame, plate_no+" 侵走非机动车道", (text_x, current_y),
                              textColor=(255, 0, 0), textSize=60)

    return frame


# ==================== 主处理函数 ====================

def process_video(video_path, gps_data, output_path=None, display=True,
                 save_output=False, conf_threshold=0.5):
    """
    处理视频文件，进行车牌检测、GPS（从JSON）和时间显示

    Args:
        video_path: 输入视频路径
        gps_data: GPS数据字典（已加载）
        output_path: 输出视频路径（可选）
        display: 是否显示实时画面
        save_output: 是否保存输出视频
        conf_threshold: 检测置信度阈值
    """
    global yolo_model, lpr_model

    # 加载模型
    if yolo_model is None or lpr_model is None:
        yolo_model, lpr_model = load_models()

    print(f"\n{'='*70}")
    print("综合视频处理系统 - 车牌识别 + GPS(从JSON) + 时间")
    print(f"{'='*70}\n")

    # 检查GPS数据
    if gps_data is None:
        print("[WARNING] GPS数据为空，将不显示GPS信息")
    elif gps_data.get('data'):
        print(f"[INFO] GPS数据已加载: {gps_data.get('total_points', 0)} 个数据点")
    else:
        print("[WARNING] GPS数据无效")

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

    # ==================== 异步推理队列 ====================
    frame_queue = queue.Queue(maxsize=5)
    latest_detections = []
    lock = threading.Lock()
    stop_flag = False

    def ai_worker():
        """后台AI推理线程"""
        nonlocal latest_detections, stop_flag

        while not stop_flag:
            try:
                item = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_id, frame = item
            detections = detect_and_recognize_plates(frame, conf_threshold)

            if detections:
                with lock:
                    latest_detections = detections

            frame_queue.task_done()

    # 启动AI线程
    t = threading.Thread(target=ai_worker, daemon=True)
    t.start()
    # =====================================================

    frame_count = 0
    all_detections = []

    start_time = time.time()

    print("\n开始处理... 按 'q' 退出，按 ' ' 暂停")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        # GPS & 时间
        lat, lon, alt = get_gps_at_time_from_json(gps_data, current_time)[:3]
        current_datetime = video_start_time + timedelta(seconds=current_time) if video_start_time else None

        # ================= 把帧丢给AI线程（不等待） =================
        if not frame_queue.full():
            frame_queue.put((frame_count, frame.copy()))
        # ============================================================

        # 取最近一次检测结果（绝不等待）
        with lock:
            detections = latest_detections.copy()

        # 保存检测结果（用于统计）
        if detections:
            for plate_no, conf, bbox in detections:
                all_detections.append({
                    'frame': frame_count,
                    'plate': plate_no,
                    'confidence': conf,
                    'bbox': bbox,
                    'time': current_time
                })

        # 画框
        frame = draw_plate_boxes(frame, detections, current_datetime, lat, lon, alt)

        # 写视频（恒速）
        if video_writer:
            video_writer.write(frame)

        # 显示进度
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%) - "
                  f"速度: {fps_actual:.1f} fps")

        # 显示画面
        if display:
            info_text = f"Frame: {frame_count}/{total_frames} | Plates: {len(detections)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("综合视频处理系统", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n处理被用户终止")
                break
            elif key == ord(' '):
                print("已暂停 - 按任意键继续...")
                cv2.waitKey(0)

    # 停止AI线程
    stop_flag = True
    t.join(timeout=1)

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
    print(f"耗时: {elapsed_time:.1f}秒")
    print(f"平均速度: {frame_count/elapsed_time:.1f} fps")
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
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='综合视频处理系统 - 车牌识别 + GPS(从JSON) + 时间',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  # 基本用法 - 使用GPS JSON文件
  python video_plate_with_json_gps.py --video clip.mov --gps clip.gps.json

  # 保存输出视频
  python video_plate_with_json_gps.py --video clip.mov --gps clip.gps.json --output result.mp4

  # 不显示画面
  python video_plate_with_json_gps.py --video clip.mov --gps clip.gps.json --no-display

注意事项:
  - GPS数据从JSON文件读取，而不是从视频元数据
  - JSON文件由trim_video.py生成
  - GPS数据包含time_offset（剪辑内的相对时间）
        """
    )

    parser.add_argument('--video', type=str, required=True,
                        help='输入视频路径（MOV格式）')
    parser.add_argument('--gps', type=str, default=None,
                        help='GPS JSON文件路径（可选，默认自动匹配）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出视频路径（可选）')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示实时画面')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='检测置信度阈值（默认: 0.5）')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.video):
        print(f"[ERROR] 找不到视频文件 '{args.video}'")
        sys.exit(1)

    # 确定 GPS JSON 文件路径
    gps_json_path = args.gps
    if gps_json_path is None:
        # 自动查找与视频同名的 .gps.json 文件
        video_path = Path(args.video)
        gps_json_path = str(video_path.with_suffix('.gps.json'))
        print(f"[INFO] 自动查找 GPS JSON: {os.path.basename(gps_json_path)}")
    else:
        print(f"[INFO] 使用指定的 GPS JSON: {gps_json_path}")

    # 检查 GPS JSON 文件
    if not os.path.exists(gps_json_path):
        print(f"[WARNING] 找不到 GPS JSON 文件 '{gps_json_path}'")
        print("[WARNING] 将不显示 GPS 信息")
        gps_data = None
    else:
        print(f"[INFO] 找到 GPS JSON 文件")
        gps_data = load_gps_from_json(gps_json_path)

    # 自动生成输出文件名
    if args.output is None:
        video_path = Path(args.video)
        output_name = f"{video_path.stem}_processed{video_path.suffix}"
        args.output = str(video_path.parent / output_name)

    # 执行处理（使用已加载的 gps_data）
    process_video(
        video_path=args.video,
        gps_data=gps_data,
        output_path=args.output,
        display=not args.no_display,
        save_output=True,
        conf_threshold=args.conf
    )
