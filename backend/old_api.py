import os
import io
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import load_model, predict_image, transform

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "backend/saved_models/best_model.pth"
model = load_model(MODEL_PATH, device)

@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert file → PIL → tensor
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    label, prob = predict_image(model, pil_img, device)

    return {
        "prediction": label,
        "confidence": round(prob * 100, 2)
    }
