import flax.linen as nn
from flax.core import FrozenDict
from jax import Array
from typing import Sequence
from jaxtyping import Array, Float


class MLP(nn.Module):
    """MLP with `len(features)` layers and ReLU activations."""

    features: Sequence[int]

    @nn.compact
    def __call__(
        self, inputs: Float[Array, "*batch data_dim"]
    ) -> Float[Array, "*batch output_dim"]:
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


class MLPWithIntermediates(nn.Module):
    """
    MLP with `len(features)` layers and ReLU activations.

    Returns intermediate activations.
    """

    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs: Float[Array, "*batch data_dim"]) -> tuple[
        Float[Array, "*batch output_dim"],
        FrozenDict[str, Float[Array, "*batch ?feature_dims"]],
    ]:
        x = inputs
        activations = {}
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
                activations[f"layer_{i}"] = x
        return x, FrozenDict(activations)
