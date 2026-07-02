import os

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

DETECTOR_WEIGHTS   = os.path.join(MODELS_DIR, "detector_best.pt")
CLASSIFIER_WEIGHTS = os.path.join(MODELS_DIR, "classifier_best.pt")

YOLO_IMGSZ   = 640
CLF_IMG_SIZE = 224
CLF_BACKBONE = "efficientnet_b0"

HAZARD_CLASSES = [
    "poison", "oxygen", "flammable", "flammable-solid", "corrosive",
    "dangerous", "non-flammable-gas", "organic-peroxide", "explosive",
    "radioactive", "inhalation-hazard", "spontaneously-combustible",
    "infectious-substance",
]

HAZARD_TO_COLOR = {
    "poison": "white", "oxygen": "yellow", "flammable": "red",
    "flammable-solid": "red", "corrosive": "white", "dangerous": "white",
    "non-flammable-gas": "green", "organic-peroxide": "red",
    "explosive": "orange", "radioactive": "yellow",
    "inhalation-hazard": "white", "spontaneously-combustible": "red",
    "infectious-substance": "white",
}

HAZARD_TO_SYMBOL = {
    "poison": "skull-crossbones", "oxygen": "flame-over-circle",
    "flammable": "flame", "flammable-solid": "flame",
    "corrosive": "corrosion", "dangerous": "exclamation",
    "non-flammable-gas": "cylinder", "organic-peroxide": "flame",
    "explosive": "explosion", "radioactive": "trefoil",
    "inhalation-hazard": "skull-crossbones",
    "spontaneously-combustible": "flame", "infectious-substance": "biohazard",
}

COLOR_CLASSES  = sorted(set(HAZARD_TO_COLOR.values()))
SYMBOL_CLASSES = sorted(set(HAZARD_TO_SYMBOL.values()))
