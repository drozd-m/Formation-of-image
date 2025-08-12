import numpy as np

# subclass_codes: (50, K)
# mine_codes: (50, K)
subclass_codes = np.load("subclass_codes.npy")  # центральні коди підкласів
mine_codes = np.load("mine_codes.npy")          # коди для зображень мін

K = subclass_codes.shape[1]
transform_matrices = []

for i in range(50):
    z_sub = subclass_codes[i]
    z_mine = mine_codes[i]

    t_diag = np.ones(K)
    mask = z_sub != 0
    t_diag[mask] = z_mine[mask] / z_sub[mask]

    T = np.diag(t_diag)
    transform_matrices.append(T)

transform_matrices = np.array(transform_matrices)  # (50, K, K)

np.save("subclass_transforms.npy", transform_matrices)
