"""
Inference + GradCAM for Deepfake Detection
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from backend.model.model import DeepfakeXceptionModel

# Transform used in training
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

CLASS_NAMES = ['fake', 'real']


# ---------------------------------------------------------
#  GRAD-CAM IMPLEMENTATION
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hook to get gradients
        target_layer.register_backward_hook(self.save_gradient)
        target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def generate(self, class_idx):
        # Average gradients over H × W
        grads = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted sum of activations
        cam = torch.sum(grads * self.activations, dim=1).squeeze()

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cam.detach().cpu().numpy()

        return cam


# ---------------------------------------------------------
#       MODEL LOADING
# ---------------------------------------------------------
def load_model(model_path: str, device: torch.device):

    model = DeepfakeXceptionModel()
    model.to(device)

    ckpt = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(ckpt)
    except:
        model.load_state_dict(ckpt["state_dict"])

    model.eval()
    return model


# ---------------------------------------------------------
#   PREDICT + GRADCAM HEATMAP
# ---------------------------------------------------------
def predict_with_gradcam(model, pil_img, device):

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Get correct last conv layer
    target_layer = model.model.conv4
    cam = GradCAM(model, target_layer)

    # Forward
    out = model(img_tensor)
    probs = F.softmax(out, dim=1)[0]
    pred_idx = int(torch.argmax(probs))

    # Backward
    model.zero_grad()
    out[0, pred_idx].backward()

    # Grad-CAM heatmap
    heatmap = cam.generate(pred_idx)

    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), heatmap


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
        # Overlay CAM heatmap
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
    parser.add_argument("--model", default="backend/saved_models/best_model.pth")
    parser.add_argument("--out", help="save output heatmap")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_image_mode(args.model, args.input, device, args.out)


if __name__ == "__main__":
    main()
