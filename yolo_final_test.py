import cv2
import torch
import numpy as np
from ultralytics import YOLO
from model.LPRNet import build_lprnet

# --- 1. 配置字符集 (必须与训练 Final_LPRNet_model.pth 时一致) ---
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
         'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         'I', 'O', '-'
         ]

# --- 2. 加载模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 YOLO 检测模型
yolo_model = YOLO("weights/best.pt")

# 加载 LPRNet 识别模型
lpr_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
lpr_model.load_state_dict(torch.load("weights/Final_LPRNet_model.pth", map_location=device))
lpr_model.to(device)
lpr_model.eval()


# --- 3. 定义推理函数 ---
def decode_res(preds, chars):
    """解码 LPRNet 输出的序列"""
    res = []
    for i in range(len(preds)):
        if preds[i] != len(chars) - 1 and (i == 0 or preds[i] != preds[i - 1]):
            res.append(chars[preds[i]])
    return "".join(res)


def process_and_recognize(img_paths):
    results = yolo_model(img_paths)

    for r in results:
        img_ori = r.orig_img.copy()  # 读取原始图
        print(f"\n图片: {r.path}")

        if len(r.boxes) == 0:
            print("  未检测到车牌")
            continue

        for i, box in enumerate(r.boxes):
            # A. 裁剪车牌 (获取坐标)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            # 适当向外扩一点点边缘，有助于识别
            crop_img = img_ori[y1:y2, x1:x2]

            # B. LPRNet 预处理 (Resize + 归一化)
            tmp_img = cv2.resize(crop_img, (94, 24))
            tmp_img = tmp_img.astype('float32')
            tmp_img -= 127.5
            tmp_img *= 0.0078125
            tmp_img = np.transpose(tmp_img, (2, 0, 1))  # HWC to CHW
            tmp_img = torch.from_numpy(tmp_img).unsqueeze(0).to(device)

            # C. LPRNet 推理
            with torch.no_grad():
                preds = lpr_model(tmp_img)
                preds = preds.cpu().numpy()
                arg_max_preds = np.argmax(preds, axis=1)  # [1, 18]
                plate_no = decode_res(arg_max_preds[0], CHARS)

            print(f"  车牌 {i + 1}: {plate_no} (置信度: {box.conf[0]:.2f})")

            # 可选：在原图画结果
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_ori, plate_no, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示带识别结果的图
        cv2.imshow("Recognition Result", img_ori)
        cv2.waitKey(0)


# --- 4. 执行 ---
source = ["test_images/1.jpg", "test_images/2.jpg", "test_images/3.jpg"]
process_and_recognize(source)
cv2.destroyAllWindows()