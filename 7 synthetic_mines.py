import torch
import torch.nn as nn
import numpy as np

# ==== 1. Модель декодера (під твої ваги) ====
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 2048)  # 64 → 2048
        self.deconv1 = nn.ConvTranspose2d(64, 128, 3)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(-1, 64, 8, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x

# ==== 2. Завантаження декодера ====
decoder = Decoder()
state_dict = torch.load("decoder.pth", map_location="cpu")
decoder.load_state_dict(state_dict, strict=False)
decoder.eval()

# ==== 3. Завантаження вхідних даних ====
central_codes = np.load("mine_codes.npy")               # 50 кодів мін (shape: [50, 64])
subclass_labels = np.load("subclass_labels.npy")        # підкласи CIFAR (shape: [50000])
subclass_transforms = np.load("subclass_transforms.npy")# 50 матриць перетворення (shape: [50, 64, 64])
all_codes = np.load("cifar_codes.npy")                  # усі коди CIFAR-10 (shape: [50000, 64])

# ==== 4. Генерація кодів мін для усіх елементів ====
generated_codes = []
for subclass_idx in range(50):
    mask = subclass_labels == subclass_idx
    subclass_codes = all_codes[mask]
    transform_matrix = subclass_transforms[subclass_idx]
    mine_code = central_codes[subclass_idx]
    
    # перетворення кодів
    transformed_codes = subclass_codes @ transform_matrix.T
    generated_codes.append(transformed_codes)

generated_codes = np.vstack(generated_codes)  
print("Форма згенерованих кодів:", generated_codes.shape)

# ==== 5. Генерація зображень ====
codes_tensor = torch.tensor(generated_codes, dtype=torch.float32)
with torch.no_grad():
    synthetic_images = decoder(codes_tensor).cpu().numpy()

# ==== 6. Збереження ====
np.save("synthetic_mines.npy", synthetic_images)
print("Збережено synthetic_mines.npy з формою:", synthetic_images.shape)
