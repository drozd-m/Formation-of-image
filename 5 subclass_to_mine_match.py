import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from scipy.spatial.distance import cdist

# --- Завантаження AutoEncoder ---
class AutoEncoder(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, K)
        )
        self.decoder = nn.Sequential(
            nn.Linear(K, 64 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

model = AutoEncoder(K=64)
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

# --- Завантаження зображень мін ---
class MineDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
mine_ds = MineDataset("mine_images", transform)
mine_dl = DataLoader(mine_ds, batch_size=32)

# --- Витягування кодів для мін ---
mine_codes = []
with torch.no_grad():
    for batch in mine_dl:
        z = model.encoder(batch)
        mine_codes.append(z.numpy())
mine_codes = np.vstack(mine_codes)  # (N_mines, 64)
np.save("mine_codes.npy", mine_codes)

# --- Завантаження центрів підкластерів CIFAR-10 ---
subclass_codes = np.load("subclass_codes.npy")  # (50, 64)

# --- Порівняння кожного підкласу з кожною міною ---
distances = cdist(subclass_codes, mine_codes, metric='euclidean')
assigned_mine_idx = np.argmin(distances, axis=1)  # (50,)

# --- Збереження відповідностей ---
np.save("subclass_to_mine_match.npy", assigned_mine_idx)

# --- Приклад виводу ---
for i, m in enumerate(assigned_mine_idx):
    print(f"Підклас {i} → міна #{m}")
