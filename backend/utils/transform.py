from torchvision import transforms

def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])
