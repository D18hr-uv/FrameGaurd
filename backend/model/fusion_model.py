# backend/model/fusion_model.py
"""
Fusion Deepfake Detector: Spatial (Xception) + FFT (Frequency CNN) branches
merged through a learned fusion classifier.

Architecture (matches fusion_best_model checkpoint):
  spatial  -> DeepfakeXceptionModel -> 2 logits
  fft      -> CNN feature extractor  -> 1 logit
  classifier -> concat(2+1=3)       -> 32 -> 16 -> 1  (sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from backend.model.model import DeepfakeXceptionModel


class FFTBranch(nn.Module):
    """
    Frequency-domain CNN branch.
    Input:  3-channel FFT magnitude spectrum image (224×224)
    Output: 1 logit (pre-sigmoid)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),   # 0
            nn.BatchNorm2d(32),                                       # 1
            nn.ReLU(inplace=True),                                    # 2
            nn.MaxPool2d(2, 2),                                       # 3

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),  # 4
            nn.BatchNorm2d(64),                                       # 5
            nn.ReLU(inplace=True),                                    # 6
            nn.MaxPool2d(2, 2),                                       # 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True), # 8
            nn.BatchNorm2d(128),                                      # 9
            nn.ReLU(inplace=True),                                    # 10
            nn.MaxPool2d(2, 2),                                       # 11

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),# 12
            nn.BatchNorm2d(256),                                      # 13
            nn.ReLU(inplace=True),                                    # 14
            nn.MaxPool2d(2, 2),                                       # 15
        )

        # 224 / 16 = 14  =>  256 * 14 * 14 = 50176
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # 0
            nn.Linear(256 * 14 * 14, 512),   # 1
            nn.ReLU(inplace=True),           # 2
            nn.Dropout(0.5),                 # 3
            nn.Linear(512, 1),               # 4
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # shape: (B, 1)


class FusionDeepfakeModel(nn.Module):
    """
    Late-fusion model combining spatial (Xception) and frequency (FFT CNN) branches.

    Forward:
        spatial_input: (B, 3, 299, 299) – regular RGB image tensor
        fft_input:     (B, 3, 224, 224) – FFT magnitude spectrum tensor

    Output:
        (B, 1) logit  – apply sigmoid for P(fake)
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # Spatial branch (Xception-based, outputs 2 logits)
        self.spatial = DeepfakeXceptionModel(num_classes=num_classes)

        # Frequency branch (CNN on FFT spectrum, outputs 1 logit)
        self.fft = FFTBranch()

        # Fusion classifier: concat spatial(2) + fft(1) = 3 → 1
        self.classifier = nn.Sequential(
            nn.Linear(3, 32),        # 0
            nn.ReLU(inplace=True),   # 1
            nn.Dropout(0.3),         # 2
            nn.Linear(32, 16),       # 3
            nn.ReLU(inplace=True),   # 4
            nn.Linear(16, 1),        # 5
        )

    def forward(self, spatial_input, fft_input):
        spatial_logits = self.spatial(spatial_input)   # (B, 2)
        fft_logit = self.fft(fft_input)                # (B, 1)
        fused = torch.cat([spatial_logits, fft_logit], dim=1)  # (B, 3)
        out = self.classifier(fused)                   # (B, 1)
        return out


# ---------------------------------------------------------------------------
#  FFT preprocessing utilities
# ---------------------------------------------------------------------------

def compute_fft_spectrum(pil_img: Image.Image, size: int = 224) -> Image.Image:
    """
    Convert a PIL RGB image to its FFT magnitude spectrum as a 3-channel
    PIL image suitable for the FFT branch.

    Steps:
        1. Resize to (size, size)
        2. Convert to grayscale numpy
        3. Apply 2D FFT, shift zero-freq to center
        4. Log-scale the magnitude
        5. Normalize to [0, 255]
        6. Stack into 3 channels (gray → RGB)
        7. Return as PIL Image
    """
    img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    gray = np.array(img.convert("L"), dtype=np.float32)

    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shifted)

    # Log scale (add 1 to avoid log(0))
    log_magnitude = np.log1p(magnitude)

    # Normalize to [0, 255]
    log_magnitude -= log_magnitude.min()
    max_val = log_magnitude.max()
    if max_val > 0:
        log_magnitude = (log_magnitude / max_val * 255.0)
    spectrum_uint8 = log_magnitude.astype(np.uint8)

    # Stack to 3-channel RGB
    spectrum_rgb = np.stack([spectrum_uint8] * 3, axis=-1)

    return Image.fromarray(spectrum_rgb, mode="RGB")


# Transforms
spatial_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

fft_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
