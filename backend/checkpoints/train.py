import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from backend.model.model import DeepfakeXceptionModel

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def load_data(dataset_path, batch_size=16):
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")

    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model():

    dataset_path = "/content/drive/MyDrive/frameguard_dataset/ffpp"
    train_loader, val_loader = load_data(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DeepfakeXceptionModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    EPOCHS = 15

    # Create save directory
    save_dir = "/content/drive/MyDrive/frameguard/backend/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {epoch_loss:.4f} "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Val Acc: {val_acc*100:.2f}%")

        # 🟢 SAVE AFTER EACH EPOCH
        save_path = f"{save_dir}/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    train_model()
