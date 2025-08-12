import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# === Параметри
K = 64
epochs = 10
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Архітектура автоенкодера
class AutoEncoder(nn.Module):
    def __init__(self, code_dim=K):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, code_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# === Завантаження CIFAR-10
transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Ініціалізація
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Навчання
model.train()
for epoch in range(epochs):
    total_loss = 0
    for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# === Збереження моделей
torch.save(model.encoder.state_dict(), "encoder.pth")
torch.save(model.decoder.state_dict(), "decoder.pth")
torch.save(model.state_dict(), "autoencoder.pth")
