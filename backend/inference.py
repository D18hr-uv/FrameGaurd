"""
Inference + GradCAM for Fusion Deepfake Detection (CLI)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from backend.model.fusion_model import (
    FusionDeepfakeModel,
    compute_fft_spectrum,
    spatial_transform,
    fft_transform,
)
from backend.model.model import DeepfakeXceptionModel
from backend.utils.gradcam import GradCAM, heatmap_on_image


# ---------------------------------------------------------
#       MODEL LOADING
# ---------------------------------------------------------
def load_model(model_path: str, device: torch.device):
    """Load fusion model from checkpoint."""
    model = FusionDeepfakeModel(num_classes=2)
    model.to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Remap fc → last_linear (xception factory renames this)
    remapped = {}
    for k, v in state_dict.items():
        new_key = k.replace("spatial.model.fc.", "spatial.model.last_linear.")
        remapped[new_key] = v

    model.load_state_dict(remapped)

    model.eval()
    return model


# ---------------------------------------------------------
#   PREDICT + GRADCAM HEATMAP
# ---------------------------------------------------------
def predict_with_gradcam(model, pil_img, device):
    """Run fusion prediction + GradCAM on spatial branch."""

    # Prepare both inputs
    spatial_input = spatial_transform(pil_img).unsqueeze(0).to(device)
    fft_spectrum = compute_fft_spectrum(pil_img, size=224)
    fft_input = fft_transform(fft_spectrum).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        fusion_logit = model(spatial_input, fft_input)
        fusion_prob = torch.sigmoid(fusion_logit).item()

    is_fake = fusion_prob > 0.5
    label = "fake" if is_fake else "real"
    confidence = fusion_prob if is_fake else (1.0 - fusion_prob)

    # GradCAM on spatial branch
    model.zero_grad()
    torch.set_grad_enabled(True)

    spatial_input_cam = spatial_transform(pil_img).unsqueeze(0).to(device).requires_grad_(True)
    target_layer = model.spatial.model.conv4
    cam = GradCAM(model.spatial, target_layer)

    try:
        spatial_out = model.spatial(spatial_input_cam)
        pred_idx = int(torch.argmax(spatial_out, dim=1).item())
        model.spatial.zero_grad()
        spatial_out[:, pred_idx].backward(retain_graph=True)
        heatmap = cam.generate_from_stored(pred_idx)
    finally:
        model.zero_grad()
        torch.set_grad_enabled(False)
        try:
            cam.remove_hooks()
        except:
            pass

    return label, confidence, heatmap


# ---------------------------------------------------------
#   IMAGE MODE
# ---------------------------------------------------------
def run_image_mode(model_path, input_path, device, out_path=None):

    model = load_model(model_path, device)
    pil = Image.open(input_path).convert("RGB")

    label, prob, heatmap = predict_with_gradcam(model, pil, device)

    print(f"\nImage: {input_path}")
    print(f"Prediction → {label} ({prob*100:.2f}%)")

    if out_path:
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_color = cv2.applyColorMap((heatmap_resized*255).astype(np.uint8), cv2.COLORMAP_JET)

        overlay = (0.35 * heatmap_color + 0.65 * img_cv).astype(np.uint8)
        cv2.putText(overlay, f"{label} ({prob*100:.2f}%)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imwrite(out_path, overlay)
        print(f"Saved GradCAM → {out_path}")


# ---------------------------------------------------------
#    MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="backend/saved_models/fusion_best_model_20260315_163859.pth")
    parser.add_argument("--out", help="save output heatmap")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_image_mode(args.model, args.input, device, args.out)


if __name__ == "__main__":
    main()
