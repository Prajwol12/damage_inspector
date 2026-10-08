from pathlib import Path

BASE = Path(__file__).parent

# Model files (to upgrade a model later, replace the file and edit here)
YOLO_PATH = BASE / "models" / "cardd_yolo26m_best.pt"
RESNET_PATH = BASE / "models" / "resnet50_best.pt"

# Damage detector (YOLO)
YOLO_IMGSZ = 832
YOLO_CONF = 0.25

# Severity classifier (ResNet50)
SEVERITY_CLASSES = ["minor", "moderate", "severe"]
SEVERITY_SIZE = (224, 288)  # (height, width) used in training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]