import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 class names (in the same order used in the confusion matrix)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DATA_DIR = "data"       # change if you used a different --data-dir
BATCH_SIZE = 16
VAL_SPLIT = 0.1
SEED = 42

def build_train_loader():
    # Same normalization as in train.py
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2626)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,      # will just reuse existing data if already downloaded
        transform=train_tf,
    )

    val_len = int(len(full_train) * VAL_SPLIT)
    train_len = len(full_train) - val_len

    train_set, _ = random_split(
        full_train,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader

def unnormalize(img_tensor):
    """Undo CIFAR-10 normalization so images look normal."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2626]).view(3, 1, 1)
    return img_tensor * std + mean

def main():
    train_loader = build_train_loader()

    images, labels = next(iter(train_loader))  # one batch from the training data

    # Unnormalize for display
    images = unnormalize(images)

    # Make a grid of images
    grid = torchvision.utils.make_grid(images, nrow=4)
    npimg = grid.numpy().transpose(1, 2, 0)
    npimg = np.clip(npimg, 0, 1)  # ensure values in [0, 1] for plotting

    plt.figure(figsize=(8, 8))
    plt.imshow(npimg)
    plt.axis("off")
    plt.title("Sample training images from CIFAR-10")
    plt.tight_layout()
    plt.show()

    # Print the labels for this batch (both indices and class names)
    print("Label indices:", labels.tolist())
    print("Label names  :", [CLASSES[i] for i in labels])

if __name__ == "__main__":
    main()
