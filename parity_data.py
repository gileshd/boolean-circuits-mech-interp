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


def sample_binary_data(key, n_samples: int, data_dim: int) -> Float[Array, "data_dim"]:
    return jr.bernoulli(key, 0.5, (n_samples, data_dim))


@partial(jit, static_argnums=(1, 2))
def sample_binary_parity_data(
    key, n_samples: int, dim: int, idx_mask: Bool[Array, "data_dim"]
) -> tuple[Bool[Array, "n_samples data_dim"], Float[Array, "n_samples 2"]]:
    """docstring"""
    x = sample_binary_data(key, n_samples, dim)
    y = vmap(parity, in_axes=(0, None))(x, idx_mask)
    return x, y


def indices_to_mask(indices: Array, data_dim) -> Bool[Array, "data_dim"]:
    return jnp.zeros(data_dim, dtype=bool).at[indices].set(True)


def make_task_bit_mask_array(
    key, n_tasks: int, n_bits_per_task: int, data_dim: int, reuse_bits=False
) -> Bool[Array, "n_tasks data_dim"]:
    """
    Creates an array of task bit masks.

    Args:
        key: The random key used for generating task bit masks.
        n_tasks (int): The number of tasks.
        n_bits_per_task (int): The number of bits per task.
        data_dim (int): The dimension of the data.

    Returns:
        jnp.ndarray: An array where the first dimension corresponds to task indices and the second dimension to the data dimension, with True values indicating task bits.
    """
    if reuse_bits:
        subkeys = jr.split(key, n_tasks)
        _choose_bit_idxs = lambda key: jr.choice(
            key, jnp.arange(data_dim), (n_bits_per_task,), replace=False
        )
        task_bit_idxs = vmap(_choose_bit_idxs)(subkeys)
    else:
        assert n_tasks * n_bits_per_task <= data_dim
        task_bit_idxs = jr.choice(
            key, jnp.arange(data_dim), (n_tasks, n_bits_per_task), replace=False
        )
    _set_task_bits = lambda idxs: jnp.zeros(data_dim, dtype=bool).at[idxs].set(True)
    task_bit_mask_array = vmap(_set_task_bits)(task_bit_idxs)
    return task_bit_mask_array


def sample_task_bits(key, n_samples, n_tasks, alpha=None):
    if alpha is not None:
        p = jnp.arange(1, n_tasks + 1) ** (-alpha)
        p /= p.sum()
        tasks = jr.choice(key, jnp.arange(n_tasks), (n_samples,), p=p)
    else:
        tasks = jr.choice(key, jnp.arange(n_tasks), (n_samples,))
    return one_hot(tasks, n_tasks)


def sample_multitask_parity_data(
    key, n_samples: int, n_tasks: int, n_bits_per_task: int, data_dim: int
) -> tuple[
    Bool[Array, "n_samples n_tasks+data_dim"],
    Float[Array, "n_samples 2"],
    Bool[Array, "n_tasks data_dim"],
]:
    """
    Sample multi-task binary parity data.

    Args:
        key: The random key used for generating task bit masks.
        n_samples (int): The number of samples.
        dim (int): The dimension of the data.
        n_tasks (int): The number of tasks.
        n_bits_per_task (int): The number of bits per task.

    Returns:
        tuple[Bool[Array, "n_samples data_dim"], Float[Array, "n_samples n_tasks 2"]]: A tuple containing the input data and the output labels.
    """
    task_bit_mask_array = make_task_bit_mask_array(
        key, n_tasks, n_bits_per_task, data_dim
    )

    key1, key2 = jr.split(key)
    task_bits = sample_task_bits(key1, n_samples, n_tasks)
    data_bits = sample_binary_data(key2, n_samples, data_dim)

    _task_parity = lambda task_idx, data: parity(data, task_bit_mask_array[task_idx])
    y = vmap(_task_parity, in_axes=(0, 0))(task_bits.argmax(1), data_bits)

    x = jnp.concatenate((task_bits, data_bits), axis=1)
    return x, y, task_bit_mask_array
