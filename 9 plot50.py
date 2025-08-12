import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 32
batch_size = 8
num_epochs = 10
lr = 1e-3


def load_images_from_folder(folder, label, image_size=32):
    images = []
    labels = []
    resize = Resize((image_size, image_size))
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            img = resize(torch.tensor(np.array(img)).permute(2, 0, 1).float())  # to CHW
            images.append(img)
            labels.append(label)
    return images, labels

mines_dir = "mines"        
no_mines_dir = "no_mines" 

mine_imgs, mine_labels = load_images_from_folder(mines_dir, 1, image_size)
no_mine_imgs, no_mine_labels = load_images_from_folder(no_mines_dir, 0, image_size)

X = torch.stack(mine_imgs + no_mine_imgs)
y = torch.tensor(mine_labels + no_mine_labels, dtype=torch.long)
X /= 255.0
# ====  Train/Test split ====
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# ====  MLP-Mixer ====
class MLPMixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, token_dim, channel_dim):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(num_patches),
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

class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_channels=3, dim=128, depth=4, token_dim=64, channel_dim=256, num_classes=2):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        self.patch_emb = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.mixer_blocks = nn.Sequential(
            *[MLPMixerBlock(num_patches, dim, token_dim, channel_dim) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

model = MLPMixer(image_size=image_size, num_channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


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


train_losses, test_losses = [], []
train_accs, test_accs = [], []
train_probs, test_probs = [], []

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


epochs = range(1, num_epochs + 1)

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
