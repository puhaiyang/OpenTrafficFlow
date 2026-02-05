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
yolo_model = YOLO("weights/best.pt")
lpr_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
lpr_model.load_state_dict(torch.load("weights/Final_LPRNet_model.pth", map_location=device))
lpr_model.to(device)
lpr_model.eval()


def cv2ImgAddText(img, text, pos, textColor=(0, 255, 0), textSize=20):
    """在图像上添加中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # 检测系统中文字体
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

    draw.text(pos, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def decode_res(preds, chars):
    """CTC Greedy 解码：去空白符、去重复"""
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


def process_and_recognize(img_paths):
    results = yolo_model(img_paths)

    for r in results:
        img_ori = r.orig_img.copy()
        print(f"\n图片: {r.path}")

        if len(r.boxes) == 0:
            print("  未检测到车牌")
            continue

        for i, box in enumerate(r.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # 裁剪车牌
            crop_img = img_ori[y1:y2, x1:x2]

            # LPRNet 预处理
            tmp_img = cv2.resize(crop_img, (94, 24))
            tmp_img = tmp_img.astype('float32')
            tmp_img -= 127.5
            tmp_img *= 0.0078125
            tmp_img = np.transpose(tmp_img, (2, 0, 1))
            tmp_img = torch.from_numpy(tmp_img).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                preds = lpr_model(tmp_img)
                preds = preds.cpu().numpy()
                arg_max_preds = np.argmax(preds, axis=1)
                plate_no = decode_res(arg_max_preds[0], CHARS)

            print(f"  车牌 {i + 1}: {plate_no} (置信度: {box.conf[0]:.2f})")

            # 绘制结果
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_ori = cv2ImgAddText(img_ori, plate_no, (x1, y1 - 25), textColor=(0, 255, 0), textSize=24)

        cv2.imshow("Recognition Result", img_ori)
        cv2.waitKey(0)


if __name__ == "__main__":
    source = ["test_images/1.jpg", "test_images/2.jpg", "test_images/3.jpg"]
    process_and_recognize(source)
    cv2.destroyAllWindows()
