import math
import numpy as np
import torch
import torch.nn as nn


class PhysicsParams(nn.Module):
    """The learnable constants of the corrosion rate law."""
    def __init__(self):
        super().__init__()
        self.log_A = nn.Parameter(torch.tensor(math.log(0.1)))  # A = exp(log_A), always positive
        self.B = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.k_s = nn.Parameter(torch.tensor(0.5))
        self.m = nn.Parameter(torch.tensor(0.5))

    def rate(self, x_real):
        """Corrosion rate in mm/yr. x_real has shape (N, 6) in REAL units."""
        pH, Cl, SO4 = x_real[:, 0], x_real[:, 1], x_real[:, 2]
        resistivity, moisture = x_real[:, 3], x_real[:, 4]

        A = torch.exp(self.log_A)
        return (
            A
            * torch.exp(-self.B * (pH - 7.0))                    # lower pH -> faster
            * torch.clamp(Cl, min=1e-6) ** self.beta              # more chloride -> faster
            * torch.clamp(1.0 + self.k_s * SO4, min=1e-6)         # sulfate -> mild accelerant
            * torch.clamp(moisture, min=1e-6) ** self.m           # wetter -> faster
            / (resistivity + 1.0)                                 # more resistive -> slower
        )


def load_stats(data_file="data.npz"):
    d = np.load(data_file)
    return {k: torch.tensor(d[k], dtype=torch.float32)
            for k in ["X_mean", "X_std", "y_mean", "y_std"]}


def pinn_loss(net, phys, X_data, y_data, X_coll, stats, lambda_phys=0.1):
    # ---- 1. data loss ----
    data_loss = torch.mean((net(X_data) - y_data) ** 2)

    # ---- 2. physics loss ----
    # the network's own slope: d(output)/d(time input)
    X_c = X_coll.clone().requires_grad_(True)
    out = net(X_c)                                              # shape (500,)
    grads = torch.autograd.grad(out.sum(), X_c, create_graph=True)[0]  # shape (500, 6)
    dnet_dt = grads[:, 5]                                       # time is column 5

    # the physics slope, converted to the same normalized units
    x_real = X_coll * stats["X_std"] + stats["X_mean"]
    rate_mm_per_yr = phys.rate(x_real)
    dphys_dt = rate_mm_per_yr * stats["X_std"][5] / stats["y_std"]

    physics_loss = torch.mean((dnet_dt - dphys_dt) ** 2)

    total = data_loss + lambda_phys * physics_loss
    return total, data_loss, physics_loss


if __name__ == "__main__":
    from model import CorrosionNet
    from collocation import sample_collocation_points

    torch.manual_seed(42)
    data = np.load("data.npz")
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32).reshape(-1)

    net = CorrosionNet()
    phys = PhysicsParams()
    X_coll = sample_collocation_points(500)
    stats = load_stats()

    total, dl, pl = pinn_loss(net, phys, X_train, y_train, X_coll, stats)
    print(f"data_loss={dl.item():.4f}  physics_loss={pl.item():.4f}  total={total.item():.4f}")
    total.backward()
    print("gradient reached network:", net.hidden.weight.grad is not None)
    print("gradient reached physics constants:", phys.B.grad is not None)