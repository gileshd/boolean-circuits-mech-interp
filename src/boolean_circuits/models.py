import flax.linen as nn
from jax import Array
from typing import Sequence
from jaxtyping import Array, Float


class MLP(nn.Module):
    """MLP with `len(features)` layers and ReLU activations."""
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs: Float[Array, "*batch data_dim"]) -> Float[Array, "*batch 2"]:
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x
