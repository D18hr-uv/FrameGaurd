# # backend/model/predict.py
# import torch
# from PIL import Image
# from torchvision import transforms
# import numpy as np
# import io
# import base64
# from pathlib import Path

# from backend.model.model import DeepfakeXceptionModel  # your model wrapper
# from backend.utils.gradcam import GradCAM, heatmap_on_image

# # same transform used for training
# transform = transforms.Compose([
#     transforms.Resize((299, 299)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# ])

# def load_model_device(model_path: str, device=None):
#     device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#     model = DeepfakeXceptionModel(num_classes=2)
#     ckpt = torch.load(model_path, map_location=device)
#     # handle both raw state_dict or dict with 'state_dict'
#     if isinstance(ckpt, dict) and 'state_dict' in ckpt:
#         model.load_state_dict(ckpt['state_dict'])
#     else:
#         model.load_state_dict(ckpt)
#     model.to(device)
#     model.eval()
#     return model, device

# def pil_to_rgb_array(pil_img: Image.Image):
#     return np.array(pil_img.convert("RGB"))

# def predict_with_gradcam(model, device, pil_img: Image.Image, target_layer=None):
#     """Return dict with label, prob, heatmap_overlay (uint8 RGB), raw_heatmap (float HxW)"""
#     img_rgb = pil_to_rgb_array(pil_img)
#     inp = transform(pil_img).unsqueeze(0).to(device)  # [1,C,H,W]
#     with torch.no_grad():
#         out = model(inp)
#         probs = torch.softmax(out, dim=1).cpu().squeeze(0).numpy()  # [2]
#     pred_idx = int(probs.argmax())
#     pred_prob = float(probs[pred_idx])

#     # For gradcam we need gradients, so run with grad enabled
#     model.zero_grad()
#     inp2 = transform(pil_img).unsqueeze(0).to(device).requires_grad_(True)
#     # choose sensible target_layer if None
#     if target_layer is None:
#         # try to pick conv4 in xception path: model.model.conv4 if available
#         try:
#             target_layer = model.model.conv4
#         except Exception:
#             # fallback: last conv in model
#             # find last nn.Conv2d / SeparableConv2d attribute
#             target_layer = None
#             for name, module in reversed(list(model.named_modules())):
#                 # choose first conv-like found
#                 from torch.nn import Conv2d
#                 if isinstance(module, Conv2d):
#                     target_layer = module
#                     break
#             if target_layer is None:
#                 raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")

#     from backend.utils.gradcam import GradCAM
#     cam = GradCAM(model, target_layer)
#     heat = cam(inp2, class_idx=pred_idx)   # float Hf x Wf [0..1]
#     overlay = heatmap_on_image(img_rgb, heat, alpha=0.45)

#     return {
#         "label": "real" if pred_idx == 1 else "fake",
#         "prob": pred_prob,
#         "heatmap": overlay,         # uint8 RGB HxW3
#         "raw_heatmap": heat         # float HxW
#     }

# def encode_image_to_base64(img_np):
#     """img_np: RGB uint8 -> base64 PNG string"""
#     import cv2, base64
#     is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
#     if not is_success:
#         raise RuntimeError("Encoding failed")
#     b64 = base64.b64encode(buffer).decode("utf-8")
#     return f"data:image/png;base64,{b64}"


# backend/model/predict.py
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import base64
import cv2

from backend.model.model import DeepfakeXceptionModel  # your model wrapper
from backend.utils.gradcam import GradCAM, heatmap_on_image

# same transform used for training
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def load_model_device(model_path: str, device=None):
    """Load model and place on device. Ensure eval() before returning."""
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = DeepfakeXceptionModel(num_classes=2)
    ckpt = torch.load(model_path, map_location=device)
    # handle both raw state_dict or dict with 'state_dict'
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()   # VERY IMPORTANT: keep in eval mode for inference
    # disable gradients by default to be safe
    torch.set_grad_enabled(False)
    return model, device


def pil_to_rgb_array(pil_img: Image.Image):
    return np.array(pil_img.convert("RGB"))


def predict_with_gradcam(model, device, pil_img: Image.Image, target_layer=None):
    """
    Return dict with:
      label: 'real'/'fake'
      prob: float (probability of chosen class)
      heatmap: overlay RGB uint8 (HxWx3)
      raw_heatmap: float HxW in [0..1]
    """
    # ensure model in eval mode (defensive)
    model.eval()

    # input image (numpy RGB) for overlay later
    img_rgb = pil_to_rgb_array(pil_img)

    # prepare tensor
    inp = transform(pil_img).unsqueeze(0).to(device)  # [1,C,H,W]

    # 1) quick forward for class probs (no grad)
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().squeeze(0).numpy()
    pred_idx = int(probs.argmax())
    pred_prob = float(probs[pred_idx])

    # 2) Grad-CAM: enable gradients only for this block
    # make sure grads are enabled and clean any previous grads
    model.zero_grad()
    torch.set_grad_enabled(True)

    # create a fresh input that requires grad (do not reuse previous tensor to avoid accidental grad accumulation)
    inp_for_cam = transform(pil_img).unsqueeze(0).to(device).requires_grad_(True)

    # choose sensible target_layer if None
    if target_layer is None:
        try:
            # common path for your wrapper: model.model.conv4
            target_layer = model.model.conv4
        except Exception:
            # fallback: find last Conv2d-like module
            target_layer = None
            from torch.nn import Conv2d
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, Conv2d):
                    target_layer = module
                    break
            if target_layer is None:
                # as final fallback just use entire model (GradCAM impl must handle)
                raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")

    # instantiate GradCAM (local instance so hooks are local)
    cam = GradCAM(model, target_layer)

    try:
        # call-gradcam: expect GradCAM to perform forward+backward internally
        heat = cam(inp_for_cam, class_idx=pred_idx)   # float HxW in [0..1]

        # overlay heatmap on original RGB (uint8)
        overlay = heatmap_on_image(img_rgb, heat, alpha=0.45)

    finally:
        # cleanup: zero grads, disable global grad, and attempt to remove hooks if implemented
        model.zero_grad()
        torch.set_grad_enabled(False)
        # If GradCAM class exposes a hook removal, attempt it safely
        try:
            if hasattr(cam, "remove_hooks"):
                cam.remove_hooks()
        except Exception:
            # ignore removal errors but keep defensive (we don't want to crash)
            pass

    return {
        "label": "real" if pred_idx == 1 else "fake",
        "prob": pred_prob,
        "heatmap": overlay,         # uint8 RGB HxW3
        "raw_heatmap": heat         # float HxW
    }


def encode_image_to_base64(img_np):
    """img_np: RGB uint8 -> base64 PNG string"""
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if not is_success:
        raise RuntimeError("Encoding failed")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"
