import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from model.LPRNet import build_lprnet
from PIL import Image, ImageDraw, ImageFont

# 配置字符集
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
         'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         'I', 'O', '-']

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

yolo_model = YOLO("weights/best.pt")
lpr_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
lpr_model.load_state_dict(torch.load("weights/Final_LPRNet_model.pth", map_location=device))
lpr_model.to(device)
lpr_model.eval()
print("Models loaded successfully!")


def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=50):
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

    draw.text(pos, text+" 侵走非机动车道", textColor, font=font)
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


def process_frame(frame, conf_threshold=0.5):
    """
    处理单帧图像，检测并识别车牌

    Args:
        frame: 输入图像帧
        conf_threshold: YOLO检测置信度阈值

    Returns:
        frame: 绘制结果的图像
        results: 检测结果列表 [(plate_no, conf, bbox), ...]
    """
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
                # 绘制结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frame = cv2ImgAddText(frame, plate_no, (x1-100, y1 - 100),
                                      textColor=(255, 0, 0), textSize=60)

                detections.append((plate_no, conf, (x1, y1, x2, y2)))

    return frame, detections


def process_video(video_path, output_path=None, display=True, save_output=False, conf_threshold=0.5):
    """
    处理视频文件，进行车牌检测与识别

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        display: 是否显示实时画面
        save_output: 是否保存输出视频
        conf_threshold: 检测置信度阈值
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {width}x{height} @ {fps}fps, Total frames: {total_frames}")

    # 创建视频写入器
    video_writer = None
    if save_output and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    frame_count = 0
    all_detections = []

    print("\nProcessing video... Press 'q' to quit early")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 处理当前帧
        processed_frame, detections = process_frame(frame, conf_threshold)

        # 保存检测结果
        for plate_no, conf, bbox in detections:
            all_detections.append({
                'frame': frame_count,
                'plate': plate_no,
                'confidence': conf,
                'bbox': bbox
            })

        # 显示进度
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

        # 写入输出视频
        if video_writer:
            video_writer.write(processed_frame)

        # 显示画面
        if display:
            # 添加帧数和检测信息
            info_text = f"Frame: {frame_count}/{total_frames} | Detected: {len(detections)}"
            cv2.putText(processed_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("License Plate Detection", processed_frame)

            # 按 'q' 退出，按 ' ' 暂停
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user")
                break
            elif key == ord(' '):
                print("Paused - Press any key to continue...")
                cv2.waitKey(0)

    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f"\nProcessing complete! Total frames processed: {frame_count}")
    print(f"Total detections: {len(all_detections)}")

    # 输出检测统计
    if all_detections:
        print("\n--- Detection Summary ---")
        plate_counts = {}
        for det in all_detections:
            plate = det['plate']
            plate_counts[plate] = plate_counts.get(plate, 0) + 1

        print("Top 10 most detected plates:")
        for plate, count in sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {plate}: {count} times")

    return all_detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Video License Plate Detection')
    parser.add_argument('--video', type=str, default='test_video.mp4',
                        help='Input video path')
    parser.add_argument('--output', type=str, default='output_video.mp4',
                        help='Output video path (set to empty to not save)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display real-time video')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Detection confidence threshold')

    args = parser.parse_args()

    # 检查输入视频是否存在
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found!")
        print("\nUsage examples:")
        print("  python video_plate_detection.py --video test.mp4")
        print("  python video_plate_detection.py --video test.mp4 --output result.mp4")
        print("  python video_plate_detection.py --video test.mp4 --no-display --conf 0.6")
    else:
        process_video(
            video_path=args.video,
            output_path=args.output if args.output else None,
            display=not args.no_display,
            save_output=bool(args.output),
            conf_threshold=args.conf
        )
