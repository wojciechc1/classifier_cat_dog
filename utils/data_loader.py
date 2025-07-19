import os
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def clean_corrupted_images(root_folder):
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError, ValueError):
                    print(f"Removing corrupted file: {path}")
                    os.remove(path)


def data_loader(train_data_dir, val_data_dir, batch_size):

    # usuwanie uszkodzonych plikow
    clean_corrupted_images(train_data_dir)
    clean_corrupted_images(val_data_dir)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),     # zmień rozmiar
        transforms.ToTensor(),             # konwersja do tensorów
        #transforms.Normalize((0.5, 0), (0.5,))  # normalizacja (dla 1 kanału użyj (0.5,), dla 3 kanałów RGB (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader