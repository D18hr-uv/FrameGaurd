import os
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        for label_name in ["real", "fake"]:
            class_dir = os.path.join(root_dir, label_name)
            label = 0 if label_name == "real" else 1

            for img in os.listdir(class_dir):
                path = os.path.join(class_dir, img)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
