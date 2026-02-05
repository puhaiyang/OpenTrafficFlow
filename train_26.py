from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="YOLO_Data/data.yaml",
    epochs=100,
    imgsz=640,
    # 1. 尝试将 batch 提高到 128
    batch=128,
    # 2. 指定 workers 为 12（打满分配给你的 CPU 核心）
    workers=12,
    # 3. 开启缓存到内存（如果你的服务器 RAM 足够，比如 >64GB）
    cache=True,
    device=0,
    project="runs/train",
    name="yolo26_ccpd",
    save=True
)

print("训练完成，最佳权重保存在runs/train/yolo26_ccpd/weights/best.pt")
