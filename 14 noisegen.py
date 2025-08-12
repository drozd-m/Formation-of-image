import numpy as np
from torchvision import datasets, transforms

# Завантаження CIFAR-10 (тільки для прикладу)
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Беремо 5000 випадкових зображень (H=32, W=32, C=3)
no_mines = []
for i in range(5000):
    img, label = dataset[i]  # label можна ігнорувати
    no_mines.append(img.numpy().transpose(1, 2, 0))  # (H, W, C)

no_mines = np.array(no_mines)
np.save("no_mines.npy", no_mines)

print("✅ Збережено no_mines.npy розміром", no_mines.shape)
