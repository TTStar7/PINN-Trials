import numpy as np
import torch


def sample_collocation_points(n_points=500, seed=0, data_file="data.npz"):
    data = np.load(data_file)
    X_mean, X_std = data["X_mean"], data["X_std"]

    # recover the real-unit training data, then find each variable's range
    X_real = data["X_train"] * X_std + X_mean
    low = X_real.min(axis=0)
    high = X_real.max(axis=0)

    # random points spread evenly between the min and max of each variable
    rng = np.random.default_rng(seed)
    X_coll_real = rng.uniform(low, high, size=(n_points, X_real.shape[1]))

    # normalize the same way as the training data
    X_coll = (X_coll_real - X_mean) / X_std
    return torch.tensor(X_coll, dtype=torch.float32)


if __name__ == "__main__":
    X_coll = sample_collocation_points()
    print("shape:", X_coll.shape)           # expect (500, 6)
    print("min per column:", X_coll.min(dim=0).values)
    print("max per column:", X_coll.max(dim=0).values)