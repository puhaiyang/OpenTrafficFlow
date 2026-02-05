from ultralytics import YOLO

model = YOLO("weights/best.pt")
# lprnet模型位置：weights/Final_LPRNet_model.pth
# 将图片路径放在一个列表中
source = ["test_images/1.jpg", "test_images/2.jpg", "test_images/3.jpg"]

# 一次性传入列表
results = model(source)

# 遍历结果并显示
for r in results:
    r.show()  # 依次弹窗显示每张图的检测结果