import cv2


def verify_yolo_label(img_path, label_path):
    # 1. 读取图片
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # 2. 读取标签
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        cls, x, y, nw, nh = map(float, line.split())

        # 3. 将归一化坐标还原为像素坐标
        # YOLO格式: x_center, y_center, width, height (均为比例)
        x_center = x * w
        y_center = y * h
        width = nw * w
        height = nh * h

        # 计算左上角和右下角坐标
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 4. 画框 (绿色，线宽2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class: {int(cls)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. 显示结果
    cv2.imshow("Verify Label", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
verify_yolo_label(
    "F:\\project\\OpenTrafficFlow\\YOLO_Data\\images\\train\\train_000000_0142792145593_87_90_245_394_448_467_442_451_259_482_263_418_446_387_0_0_31_24_5_29_26_124_59.jpg",
    "F:\\project\\OpenTrafficFlow\\YOLO_Data\\labels\\train\\train_000000_0142792145593_87_90_245_394_448_467_442_451_259_482_263_418_446_387_0_0_31_24_5_29_26_124_59.txt")
