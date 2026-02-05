from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="YOLO_Data/data.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device=0,
    project="runs/train",
    name="yolo26_ccpd",
    save=True
)

print("训练完成，最佳权重保存在runs/train/yolo26_ccpd/weights/best.pt")
