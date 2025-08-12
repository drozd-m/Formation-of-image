import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ==== 1. Завантаження даних ====
train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")
train_images = train_images.transpose(0, 3, 1, 2)
test_images = test_images.transpose(0, 3, 1, 2)
train_dataset = TensorDataset(torch.tensor(train_images, dtype=torch.float32),
                               torch.tensor(train_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(test_images, dtype=torch.float32),
                              torch.tensor(test_labels, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==== 2. MLP-Mixer ====
class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MLPMixer(nn.Module):
    def __init__(self, dim=128, patches=64, num_classes=10):
      
        super().__init__()
        self.embed = nn.Conv2d(3, dim, 4, 4)  # (B, dim, H/4, W/4)
        self.mixers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                MLPBlock(patches),
                nn.LayerNorm(dim),
                MLPBlock(dim)
            ) for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Flatten(),
            nn.Linear(patches * dim, num_classes)
        )

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)  # (B, patches, dim)
        for norm1, token, norm2, channel in self.mixers:
            x = x + token(norm1(x).transpose(1, 2)).transpose(1, 2)
            x = x + channel(norm2(x))
        return self.head(x)

# ==== 3. Навчання ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPMixer(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, test_losses = [], []
train_accs, test_accs = [], []
train_confidences, test_confidences = [], []

epochs = 10
softmax = nn.Softmax(dim=1)

for epoch in range(epochs):
    # --- Train ---
    model.train()
    correct, total, running_loss = 0, 0, 0
    conf_sum, conf_count = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = softmax(outputs)
        _, predicted = torch.max(probs, 1)
        correct_mask = predicted == labels
        conf_sum += probs[torch.arange(len(labels)), predicted][correct_mask].sum().item()
        conf_count += correct_mask.sum().item()

        correct += correct_mask.sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)
    train_confidences.append(conf_sum / max(conf_count, 1))

    # --- Test ---
    model.eval()
    correct, total, running_loss = 0, 0, 0
    conf_sum, conf_count = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probs = softmax(outputs)
            _, predicted = torch.max(probs, 1)
            correct_mask = predicted == labels
            conf_sum += probs[torch.arange(len(labels)), predicted][correct_mask].sum().item()
            conf_count += correct_mask.sum().item()

            correct += correct_mask.sum().item()
            total += labels.size(0)

    test_losses.append(running_loss / len(test_loader))
    test_accs.append(correct / total)
    test_confidences.append(conf_sum / max(conf_count, 1))

    print(f"Epoch {epoch+1}/{epochs} "
          f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f} "
          f"Train Acc: {train_accs[-1]*100:.2f}%, Test Acc: {test_accs[-1]*100:.2f}% "
          f"Train Conf: {train_confidences[-1]:.3f}, Test Conf: {test_confidences[-1]:.3f}")

# ==== 4. Побудова графіків ====
plt.figure(figsize=(12,5))

# Loss curves
plt.subplot(1,3,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curves (Loss)")

# Accuracy curves
plt.subplot(1,3,2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curves (Accuracy)")

# Confidence curves
plt.subplot(1,3,3)
plt.plot(train_confidences, label="Train Confidence")
plt.plot(test_confidences, label="Test Confidence")
plt.xlabel("Epoch")
plt.ylabel("Avg Confidence (Correct preds)")
plt.legend()
plt.title("Confidence Curves")

plt.tight_layout()
plt.show()
