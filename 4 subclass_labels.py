import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from tqdm import tqdm

# === Параметри
L = 5  # підкласи на 1 клас
K = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Завантаження кодів та міток
cifar_codes = np.load("cifar_codes.npy")  # (50000, 64)
cifar_labels = np.load("cifar10_labels.npy")  # (50000,)

subclass_labels = np.zeros_like(cifar_labels)  # (50000,)
offset = 0

for class_id in range(10):
    # Вибираємо коди з певного класу
    idx = np.where(cifar_labels == class_id)[0]
    class_codes = cifar_codes[idx]

    # KMeans кластеризація на L підкласи
    kmeans = KMeans(n_clusters=L, random_state=42)
    cluster_ids = kmeans.fit_predict(class_codes)

    # Унікальний підкласний ID: class_id * L + cluster_id
    subclass_labels[idx] = cluster_ids + class_id * L

np.save("subclass_labels.npy", subclass_labels)
