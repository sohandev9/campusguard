from ultralytics import YOLO

model = model = YOLO(r"D:\hackathon\campusguard\models\yolov8n.pt")  # auto-downloads on first run

results = model.track(
    source="../../data/test_video.avi",
    tracker="botsort.yaml",
    persist=True,
    classes=[0],   # person only
    save=True,      # saves annotated output video
    conf=0.4
)