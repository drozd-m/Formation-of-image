import numpy as np

# Завантаження даних
features = np.load("cifar10_features.npy")           # (50000, 64)
subclass_labels = np.load("subclass_labels.npy")     # (50000,)
transforms = np.load("subclass_transforms.npy")      # (50, 64, 64)

# Ініціалізація масиву перетворених ознак
transformed_codes = np.zeros_like(features)          # (50000, 64)

# Застосування трансформацій
for i in range(len(features)):
    subclass_id = subclass_labels[i]
    W = transforms[subclass_id]
    transformed_codes[i] = W @ features[i]

# Збереження результату
np.save("transformed_codes.npy", transformed_codes)
print("✅ Збережено у transformed_codes.npy. Форма:", transformed_codes.shape)
