from functools import partial
from jax import Array, jit, vmap
from jax import numpy as jnp
from jax import random as jr
from jax.nn import one_hot
from jaxtyping import Array, Float, Bool


def parity(
    x: Float[Array, "data_dim"], idx_mask: Bool[Array, "data_dim"]
) -> Float[Array, "2"]:
    p = jnp.sum(x, where=idx_mask) % 2
    return one_hot(p, 2)


def sample_binary_data(
    key, num_samples: int, data_dim: int
) -> Float[Array, "data_dim"]:
    return jnp.array(jr.bernoulli(key, 0.5, (num_samples, data_dim)))


@partial(jit, static_argnums=(1,2))
def sample_binary_parity_data(
    key, num_samples: int, dim: int, idx_mask: Bool[Array, "data_dim"]
) -> tuple[Bool[Array, "num_samples data_dim"], Float[Array, "num_samples 2"]]:
    """docstring"""
    x = sample_binary_data(key, num_samples, dim)
    y = vmap(parity, in_axes=(0, None))(x, idx_mask)
    return x, y

