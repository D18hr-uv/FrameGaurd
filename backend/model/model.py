import torch
import torch.nn as nn
from backend.model.xception import xception   # uses the xception.py you copied earlier

class DeepfakeXceptionModel(nn.Module):
    """
    Clean deepfake classifier using Xception backbone.
    Fully compatible with your dataset:
    - images in backend/datasets/real and backend/datasets/fake
    - simple 2-class output
    """

    def __init__(self, num_classes=2):
        super(DeepfakeXceptionModel, self).__init__()

        # Load Xception from the file you copied from adversarial repo
        self.model = xception(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
