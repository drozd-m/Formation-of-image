import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

# === Параметри
K = 64
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Сумісний кодер
class Encoder(nn.Module):
    def __init__(self, code_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # -> [32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> [64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> [128, 4, 4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, code_dim)  # -> [64]
        )

    def forward(self, x):
        return self.encoder(x)
# === Завантаження моделі
encoder = Encoder().to(device)
encoder.encoder.load_state_dict(torch.load("encoder.pth", map_location=device))  # <- Зверни увагу!
encoder.eval()

# === Завантаження CIFAR-10
transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# === Кодування зображень
all_codes = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Кодування CIFAR-10"):
        images = images.to(device)
        codes = encoder(images).cpu().numpy()
        all_codes.append(codes)
        all_labels.append(labels.numpy())

# === Збереження
cifar_codes = np.concatenate(all_codes, axis=0)  # (50000, 64)
cifar_labels = np.concatenate(all_labels, axis=0)  # (50000,)

np.save("cifar_codes.npy", cifar_codes)
np.save("cifar10_labels.npy", cifar_labels)
