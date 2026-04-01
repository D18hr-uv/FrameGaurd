# backend/api/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from backend.model.predict import load_model_device, predict_with_gradcam, encode_image_to_base64

MODEL_PATH = "backend/saved_models/fusion_best_model_20260315_163859.pth"

app = FastAPI(title="FrameGuard API")

# Enable CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
@app.on_event("startup")
def startup_event():
    global MODEL, DEVICE
    MODEL, DEVICE = load_model_device(MODEL_PATH)
    print("Fusion model loaded, device:", DEVICE)

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # basic checks
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")
    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    # run prediction + gradcam
    try:
        res = predict_with_gradcam(MODEL, DEVICE, pil)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # encode overlay heatmap as base64 PNG
    heatmap_b64 = encode_image_to_base64(res["heatmap"])

    return JSONResponse({
        "label": res["label"],
        "prob": res["prob"],
        "heatmap": heatmap_b64,
        "fusion_score": res["fusion_score"],
        "spatial_fake_conf": res["spatial_fake_conf"],
        "fft_fake_conf": res["fft_fake_conf"],
    })
