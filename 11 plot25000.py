import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# --- 1. Загрузка данных ---
mine = np.load("augmented_mine_class25000.npy")         # (25000, 3, 32, 32)
no_mine = np.load("no_mines25000.npy")             # (25000, 32, 32, 3)
no_mine = no_mine.transpose(0, 3, 1, 2)      # (25000, 3, 32, 32)

# --- 2. Объединение и метки ---
X = np.concatenate([mine, no_mine], axis=0)  # (50000, 3, 32, 32)
y = np.concatenate([np.ones(len(mine)), np.zeros(len(no_mine))])

# --- 3. Преобразование в тензоры и нормализация ---
X = torch.tensor(X, dtype=torch.float32) / 255.0
y = torch.tensor(y, dtype=torch.long)

# --- 4. Создание датасета и загрузчиков ---
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# --- 5. MLP-Mixer модель ---
class MLPMixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, token_dim, channel_dim):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patches),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, hidden_dim),
        )
        
    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(x)
        return x
class MLPMixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, token_dim, channel_dim):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(num_patches),  # Нормализация по патчам
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patches),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),   # Нормализация по каналам
            nn.Linear(hidden_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, hidden_dim),
        )

    def forward(self, x):
        # Token mixing
        y = x.transpose(1, 2)  # [B, num_patches, hidden_dim] → [B, hidden_dim, num_patches]
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # Обратно
        x = x + y

        # Channel mixing
        x = x + self.channel_mlp(x)
        return x

# --- 6. Обучение ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPMixer(image_size=32, num_channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
train_losses, test_losses = [], []
train_accs, test_accs = [], []
train_probs, test_probs = [], []

def calc_accuracy_and_probs(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    probs_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            probs_list.append(correct_probs.cpu().numpy())
    acc = correct / total
    avg_prob = np.concatenate(probs_list).mean()
    return acc, avg_prob

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    train_acc, train_prob = calc_accuracy_and_probs(model, train_loader)
    train_accs.append(train_acc)
    train_probs.append(train_prob)

    test_acc, test_prob = calc_accuracy_and_probs(model, test_loader)

    model.eval()
    running_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    test_loss = running_loss / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    test_probs.append(test_prob)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Prob: {train_prob:.4f} | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Prob: {test_prob:.4f}")

# --- 7. Построение графиков ---
epochs = range(1, num_epochs+1)
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accs, label='Train Accuracy')
plt.plot(epochs, test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, train_probs, label='Train Correct Class Probability')
plt.plot(epochs, test_probs, label='Test Correct Class Probability')
plt.xlabel('Epoch')
plt.ylabel('Probability')
plt.legend()

plt.tight_layout()
plt.show()
