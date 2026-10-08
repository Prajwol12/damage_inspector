import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image, ImageOps

try:
    import pi_heif
    pi_heif.register_heif_opener()  # enables .heic / .heif (iPhone photos)
except ImportError:
    pass

MAX_SIDE = 2000  # shrink huge photos so the app stays fast and light on memory


def load_image(file):
    """Open any common image file and return a clean RGB PIL image."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)  # fix sideways phone photos

    if img.mode in ("RGBA", "LA", "P"):  # transparency -> white background
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """Load both models once. Returns (yolo, resnet)."""
    yolo = YOLO(str(config.YOLO_PATH))

    resnet = models.resnet50()
    resnet.fc = nn.Linear(2048, len(config.SEVERITY_CLASSES))
    state = torch.load(config.RESNET_PATH, map_location="cpu")
    resnet.load_state_dict(state)
    resnet.to(DEVICE).eval()
    return yolo, resnet


_preprocess = transforms.Compose([
    transforms.Resize(config.SEVERITY_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


def analyze(image: Image.Image, yolo, resnet, conf=None):
    """Run both models on one PIL image and return a results dict."""
    image = image.convert("RGB")

    # 1) Where is the damage, and what type?
    result = yolo.predict(
        image,
        imgsz=config.YOLO_IMGSZ,
        conf=conf or config.YOLO_CONF,
        verbose=False,
    )[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "type": result.names[cls_id],
            "confidence": float(box.conf[0]),
            "box": [float(v) for v in box.xyxy[0]],
        })

    annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR -> RGB

    # 2) How severe is it overall?
    x = _preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(resnet(x), dim=1)[0].cpu().tolist()

    best = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "detections": detections,
        "annotated_image": annotated,
        "severity": config.SEVERITY_CLASSES[best],
        "severity_probs": dict(zip(config.SEVERITY_CLASSES, probs)),
    }