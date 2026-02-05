from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("test_images/1.jpg")
# results = model("test_images/2.jpg")
results[0].show()
