# HazmatAnalyzer

Real-time hazardous material placard detection and classification using a two-stage deep learning pipeline.

## Architecture

The proposed system processes input from images, videos, or live camera streams to perform automated hazard placard recognition. In the first stage, a YOLOv8n-based detection network is used to localize hazardous material placards within the scene. The detector is trained on 13 hazard classes and achieves an mAP@50 score of 0.987, enabling accurate real-time placard detection.

The detected regions of interest (ROIs) are then passed to a second-stage EfficientNet-B0-based multi-head classifier. This classifier performs structured semantic understanding by simultaneously predicting three attributes for each detected placard: placard color (5 categories), and symbol type (9 categories). Finally, the system generates a structured JSON output containing the predicted semantic attributes and confidence scores for each detected hazard placard.

**Models**
| Model | Architecture | Parameters | Purpose |
|-------|-------------|------------|---------|
| `detector_best.pt` | YOLOv8n | 3.2M | Placard localisation |
| `classifier_best.pt` | EfficientNet-B0 | 5.3M | Attribute classification |

**Hazard Classes**
`poison` · `oxygen` · `flammable` · `flammable-solid` · `corrosive` · `dangerous` · `non-flammable-gas` · `organic-peroxide` · `explosive` · `radioactive` · `inhalation-hazard` · `spontaneously-combustible` · `infectious-substance`

---

## Setup

```bash
conda activate torch_env
pip install ultralytics timm opencv-python torchvision
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA optional (CPU inference supported)

---

## Usage

### Single Image
```bash
python infer.py image path/to/image.jpg
python infer.py image path/to/image.jpg --viz   # saves annotated image
```

Output JSON:
```json
{
  "image": "image.jpg",
  "num_detections": 2,
  "detections": [
    {
      "bounding_box": [120, 45, 380, 310],
      "det_confidence": 0.934,
      "hazard_class": "flammable",
      "color": "red",
      "symbol": "flame",
      "confidence": 0.891,
      "conf_class": 0.923,
      "conf_color": 0.976,
      "conf_symbol": 0.912
    }
  ],
  "timing_ms": { "detection": 18.4, "classification": 6.2, "total": 25.1 }
}
```

### Video File
```bash
python infer.py video path/to/video.mp4
python infer.py video path/to/video.mp4 --out result.mp4 --skip 2
```
`--skip N` runs inference every N frames (keeps display smooth on long videos).

### Live Camera
```bash
python infer.py camera
python infer.py camera --camera 1 --skip 2
```

**Camera controls**

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Save screenshot to `captures/` |
| `p` | Pause / unpause |
| `+` / `-` | Increase / decrease frame skip |

---

## Project Structure

```
HazmatAnalyzer/
├── infer.py                      # Inference entry point
├── models/
│   ├── detector_best.pt          # YOLOv8n weights
│   └── classifier_best.pt        # EfficientNet-B0 weights
├── configs/
│   └── config.py                 # Paths and class labels
├── src/
│   ├── pipeline.py               # End-to-end pipeline
│   ├── detection/
│   │   └── detector.py           # YOLO wrapper
│   └── classification/
│       ├── model.py              # Multi-head model definition
│       └── classifier.py         # Classifier wrapper
└── captures/                     # Screenshots saved here
```

---

## Performance

| Metric | Value |
|--------|-------|
| Detection mAP50 | 0.987 |
| Classification accuracy (hazard class) | ~95% |
| Classification accuracy (color) | ~97% |
| Classification accuracy (symbol) | ~96% |
| End-to-end latency (RTX 2050) | ~25 ms |
| End-to-end latency (CPU) | ~180 ms |
