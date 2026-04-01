# # backend/utils/gradcam.py
# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2

# class GradCAM:
#     """
#     Simple Grad-CAM for PyTorch.
#     Usage:
#       cam = GradCAM(model, target_layer)   # target_layer: nn.Module
#       heatmap = cam(pil_tensor, class_idx) # returns heatmap (H,W) float32 [0..1]
#     """
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         # register hooks
#         def forward_hook(module, input, output):
#             self.activations = output.detach()
#         def backward_hook(module, grad_in, grad_out):
#             # grad_out is a tuple
#             self.gradients = grad_out[0].detach()
#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_backward_hook(backward_hook)

#     def __call__(self, input_tensor, class_idx=None):
#         """
#         input_tensor: torch.Tensor shape [1,C,H,W], on same device as model
#         class_idx: index of target class (0..num_classes-1). If None, uses highest logit.
#         Returns: heatmap numpy HxW float32 in [0,1]
#         """
#         self.model.zero_grad()
#         out = self.model(input_tensor)        # forward
#         if class_idx is None:
#             class_idx = int(torch.argmax(out, dim=1).item())
#         score = out[:, class_idx]
#         score.backward(retain_graph=True)

#         # gradients: [N, C, Hf, Wf]; activations: [N, C, Hf, Wf]
#         gradients = self.gradients[0]   # C, Hf, Wf
#         activations = self.activations[0]  # C, Hf, Wf

#         # global average pool grads
#         weights = torch.mean(gradients, dim=(1, 2))  # C
#         # weighted sum of activations
#         cam = torch.zeros(activations.shape[1:], dtype=activations.dtype, device=activations.device)
#         for i, w in enumerate(weights):
#             cam += w * activations[i]

#         cam = torch.relu(cam)
#         cam = cam.cpu().numpy()
#         # resize to input size
#         cam -= cam.min()
#         if cam.max() != 0:
#             cam = cam / cam.max()
#         return cam.astype(np.float32)

# def heatmap_on_image(img_rgb: np.ndarray, heatmap: np.ndarray, alpha=0.4, colormap=cv2.COLORMAP_JET):
#     """
#     img_rgb: HxW x 3 uint8 (RGB)
#     heatmap: HxW float32 [0..1]
#     returns overlay RGB uint8
#     """
#     h, w = img_rgb.shape[:2]
#     heat = cv2.resize((heatmap * 255).astype(np.uint8), (w, h))
#     heat_color = cv2.applyColorMap(heat, colormap)  # BGR
#     heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
#     overlay = cv2.addWeighted(img_rgb.astype(np.float32), 1.0, heat_color.astype(np.float32), alpha, 0)
#     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#     return overlay


# backend/utils/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Simple Grad-CAM for PyTorch with proper hook cleanup.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []  # NEW: store hook handles

        # register hooks
        self.handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self.handles.append(
            target_layer.register_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        """Remove all hooks safely."""
        for h in self.handles:
            try:
                h.remove()
            except:
                pass
        self.handles = []

    def generate_from_stored(self, class_idx=None):
        """
        Generate heatmap from already-stored activations and gradients.
        Use this when forward+backward is performed externally.
        """
        gradients = self.gradients[0]     # C, Hf, Wf
        activations = self.activations[0] # C, Hf, Wf

        weights = torch.mean(gradients, dim=(1, 2))  # C
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype, device=activations.device)

        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = torch.relu(cam).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.astype(np.float32)

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: [1,C,H,W]
        class_idx: target class index
        Returns heatmap (HxW float32)
        """
        self.model.zero_grad()
        out = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(out, dim=1).item())

        score = out[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]     # C, Hf, Wf
        activations = self.activations[0] # C, Hf, Wf

        # Grad-CAM core
        weights = torch.mean(gradients, dim=(1, 2))  # C
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype, device=activations.device)

        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = torch.relu(cam).cpu().numpy()

        # normalize
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.astype(np.float32)


def heatmap_on_image(img_rgb: np.ndarray, heatmap: np.ndarray, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image.
    img_rgb: HxWx3 RGB uint8
    heatmap: HxW float32 [0..1]
    """
    h, w = img_rgb.shape[:2]
    heat = cv2.resize((heatmap * 255).astype(np.uint8), (w, h))
    heat_color = cv2.applyColorMap(heat, colormap)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb.astype(np.float32), 1.0,
                              heat_color.astype(np.float32), alpha, 0)
    return np.clip(overlay, 0, 255).astype(np.uint8)
