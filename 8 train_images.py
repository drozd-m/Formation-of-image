import numpy as np
from sklearn.model_selection import train_test_split

mines = np.load("synthetic_mines.npy")        # (50000, 3, 32, 32)
x_resized = np.zeros((50000, 3, 32, 32), dtype=np.float32)
for i in range(50000):
    for c in range(3):
        x_resized[i, c] = cv2.resize(mines[i, c], (32, 32), interpolation=cv2.INTER_LINEAR)
x_new = np.transpose(x_resized, (0, 2, 3, 1))
print(x_resized.shape)
no_mines = np.load("no_mines.npy")  # (5000, 3, 32, 32)

print("Mines:", mines.shape)
print("No mines:", no_mines.shape)

#9 підкласів по 5000 зображень
mine_subclasses = np.array_split(x_new, 9)

train_images = []
train_labels = []
test_images = []
test_labels = []

# Для 9 класів с мінами
for i, subclass in enumerate(mine_subclasses):
    X_train, X_test = train_test_split(subclass, test_size=0.2, random_state=42)
    train_images.append(X_train)
    test_images.append(X_test)
    train_labels.append(np.full(len(X_train), i))       # Класс i
    test_labels.append(np.full(len(X_test), i))

# Для класу без мін
X_train, X_test = train_test_split(no_mines, test_size=0.2, random_state=42)
train_images.append(X_train)
test_images.append(X_test)
train_labels.append(np.full(len(X_train), 9))  # Класс 9 — без мин
test_labels.append(np.full(len(X_test), 9))


train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

print("Train set:", train_images.shape, train_labels.shape)
print("Test set:", test_images.shape, test_labels.shape)


np.save("train_images.npy", train_images)
np.save("train_labels.npy", train_labels)
np.save("test_images.npy", test_images)
np.save("test_labels.npy", test_labels)
