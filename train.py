import numpy as np
import torch
import torch.nn as nn
from model import CorrosionNet

torch.manual_seed(42)

data = np.load("data.npz")
X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32).reshape(-1)
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"], dtype=torch.float32).reshape(-1)

net = CorrosionNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

n_epochs = 2000
patience = 200                  # stop if test loss hasn't improved for this many epochs
best_test = float("inf")
best_epoch = 0

for epoch in range(n_epochs):
    # train step
    train_loss = loss_fn(net(X_train), y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # check on unseen sites (no gradients needed)
    with torch.no_grad():
        test_loss = loss_fn(net(X_test), y_test)

    # remember the best weights so far
    if test_loss.item() < best_test:
        best_test = test_loss.item()
        best_epoch = epoch
        torch.save(net.state_dict(), "corrosion_net.pt")

    if epoch % 100 == 0:
        print(f"epoch {epoch:4d}  train {train_loss.item():.4f}  test {test_loss.item():.4f}")

    # early stopping
    if epoch - best_epoch > patience:
        print(f"stopping early at epoch {epoch}")
        break

print(f"best test loss {best_test:.4f} at epoch {best_epoch}")