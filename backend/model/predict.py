# backend/model/predict.py
"""
Prediction + GradCAM for the Fusion Deepfake Detection model.
Handles the spatial+FFT fusion architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
import base64
import cv2
from PIL import Image

from backend.model.fusion_model import (
    FusionDeepfakeModel,
    compute_fft_spectrum,
    spatial_transform,
    fft_transform,
)
from backend.utils.gradcam import GradCAM, heatmap_on_image


def load_model_device(model_path: str, device=None):
    """Load fusion model and place on device."""
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = FusionDeepfakeModel(num_classes=2)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and wrapped dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Remap checkpoint keys: the fusion checkpoint was saved with spatial.model.fc.*
    # but the xception() factory renames fc → last_linear
    remapped = {}
    for k, v in state_dict.items():
        new_key = k.replace("spatial.model.fc.", "spatial.model.last_linear.")
        remapped[new_key] = v
    state_dict = remapped

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    return model, device


def pil_to_rgb_array(pil_img: Image.Image):
    return np.array(pil_img.convert("RGB"))


def predict_with_gradcam(model, device, pil_img: Image.Image, target_layer=None):
    """
    Run fusion model prediction + Grad-CAM on the spatial branch.

    Returns dict with:
        label:       'real' or 'fake'
        prob:        float confidence (0..1)
        heatmap:     overlay RGB uint8 (HxWx3)
        raw_heatmap: float HxW in [0..1]
        spatial_conf: spatial branch confidence detail
        fft_conf:     fft branch confidence detail
    """
    model.eval()

    img_rgb = pil_to_rgb_array(pil_img)

    # ---- Prepare both inputs ----
    spatial_input = spatial_transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 299, 299)

    fft_spectrum = compute_fft_spectrum(pil_img, size=224)
    fft_input = fft_transform(fft_spectrum).unsqueeze(0).to(device)     # (1, 3, 224, 224)

    # ---- Forward pass (no grad) for predictions ----
    with torch.no_grad():
        # Full fusion output
        fusion_logit = model(spatial_input, fft_input)  # (1, 1)
        fusion_prob = torch.sigmoid(fusion_logit).item()  # P(fake)

        # Individual branch outputs for detail display
        spatial_logits = model.spatial(spatial_input)  # (1, 2)
        spatial_probs = torch.softmax(spatial_logits, dim=1).cpu().squeeze(0).numpy()
        spatial_fake_prob = float(spatial_probs[0])  # idx 0 = fake

        fft_logit = model.fft(fft_input)  # (1, 1)
        fft_prob = torch.sigmoid(fft_logit).item()

    # Determine label: fusion_prob > 0.5 = fake
    is_fake = fusion_prob > 0.5
    label = "fake" if is_fake else "real"
    confidence = fusion_prob if is_fake else (1.0 - fusion_prob)

    # ---- Grad-CAM on spatial branch ----
    model.zero_grad()
    torch.set_grad_enabled(True)

    spatial_input_cam = spatial_transform(pil_img).unsqueeze(0).to(device).requires_grad_(True)
    fft_input_cam = fft_transform(fft_spectrum).unsqueeze(0).to(device)

    # Target: last conv in the Xception backbone (spatial.model.conv4)
    if target_layer is None:
        try:
            target_layer = model.spatial.model.conv4
        except AttributeError:
            # Fallback: find last Conv2d
            from torch.nn import Conv2d
            for name, module in reversed(list(model.spatial.named_modules())):
                if isinstance(module, Conv2d):
                    target_layer = module
                    break
            if target_layer is None:
                raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")

    cam = GradCAM(model.spatial, target_layer)

    try:
        # Run spatial branch forward for Grad-CAM
        # Use spatial logits and backprop on the predicted class
        spatial_out = model.spatial(spatial_input_cam)
        pred_idx = int(torch.argmax(spatial_out, dim=1).item())

        model.spatial.zero_grad()
        score = spatial_out[:, pred_idx]
        score.backward(retain_graph=True)

        heat = cam.generate_from_stored(pred_idx)
        overlay = heatmap_on_image(img_rgb, heat, alpha=0.45)

    finally:
        model.zero_grad()
        torch.set_grad_enabled(False)
        try:
            cam.remove_hooks()
        except Exception:
            pass

    return {
        "label": label,
        "prob": confidence,
        "heatmap": overlay,
        "raw_heatmap": heat,
        "fusion_score": round(fusion_prob, 4),
        "spatial_fake_conf": round(spatial_fake_prob, 4),
        "fft_fake_conf": round(fft_prob, 4),
    }


def encode_image_to_base64(img_np):
    """img_np: RGB uint8 -> base64 PNG data URL"""
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if not is_success:
        raise RuntimeError("Encoding failed")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"
