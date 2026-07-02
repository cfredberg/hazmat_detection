from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os

from HazmatAnalyzer.src.pipeline import HazmatPipeline

app = FastAPI()

# allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load your ML model once
pipe = HazmatPipeline(det_conf=0.25)

# convert hazard names → class numbers
CLASS_MAP = {
    "explosive": 1,
    "non-flammable-gas": 2,
    "oxygen": 2,
    "flammable": 3,
    "flammable-solid": 4,
    "organic-peroxide": 5,
    "poison": 6,
    "inhalation-hazard": 6,
    "infectious-substance": 6,
    "radioactive": 7,
    "corrosive": 8,
    "dangerous": 9,
    "spontaneously-combustible": 4,
}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = tmp.name

    try:
        # run your model
        output = pipe.run(temp_path)

        detections = output.get("detections", [])

        if len(detections) == 0:
            return {"detected": False}

        # take best detection
        best = max(detections, key=lambda x: x["confidence"])

        hazard_name = best["hazard_class"]
        hazard_name = hazard_name.lower()

        class_num = CLASS_MAP.get(hazard_name)

        return {
            "detected": True,
            "class": class_num,
            "hazard": hazard_name,
            "confidence": best["confidence"]
        }

    finally:
        os.remove(temp_path)