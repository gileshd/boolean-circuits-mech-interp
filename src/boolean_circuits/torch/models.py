import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_input: int, d_mlp: int, n_hidden_layers: int, d_output: int = 1, hooked: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d_input, d_mlp)] + [nn.Linear(d_mlp, d_mlp) for _ in range(n_hidden_layers)]
        )
        self.fc = nn.Linear(d_mlp, d_output)
        if hooked:
            self.activations = {}

    def forward(self, x: torch.Tensor):
        if self.hooked: self.activations.clear()
        for i, layer in enumerate(self.mlp.layers):
            x = layer(x)
            x = torch.relu(x)
            if self.hooked: self.activations[f'layer_{i}'] = x
        x = self.mlp.fc(x)
        if self.hooked: self.activations['output'] = x
        return torch.sigmoid(x)