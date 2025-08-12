import numpy as np
NUM_SUBCLASSES = 50
FEATURE_DIM = 64  # розмірність ознак (K)
# Генерація випадкових центрів підкластерів
np.random.seed(42)
subclass_codes = np.random.rand(NUM_SUBCLASSES, FEATURE_DIM)
# Збереження у .npy
np.save("subclass_codes.npy", subclass_codes)
