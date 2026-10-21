import torch
import torch.nn as nn

class CorrosionNet(nn.Module):
    def __init__(self, n_inputs=6, n_hidden=8):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        return self.out(h).squeeze(-1)