from jax import Array, vmap
from jax import numpy as jnp
from jax import random as jr
from jax.nn import one_hot
from jaxtyping import Array, Float, Bool


def parity(x: Bool[Array, "data_dim"], idx_mask: Bool[Array, "data_dim"]) -> Bool:
    return jnp.sum(x, where=idx_mask) % 2


def one_hot_parity(
    x: Float[Array, "data_dim"], idx_mask: Bool[Array, "data_dim"]
) -> Float[Array, "2"]:
    p = parity(x, idx_mask)
    return one_hot(p, 2)


def sample_binary_data(key, n_samples: int, data_dim: int) -> Float[Array, "data_dim"]:
    return jr.bernoulli(key, 0.5, (n_samples, data_dim))


def sample_binary_parity_data(
    key, n_samples: int, dim: int, idx_mask: Bool[Array, "dim"]
) -> tuple[Bool[Array, "n_samples dim"], Float[Array, "n_samples 2"]]:
    """
    Sample sparse parity data.

    Args:
        key: PRNG key for sampling.
        n_samples: The number of samples to generate.
        dim: The dimension of the data.
        idx_mask: The mask for selecting indices.

    Returns:
        A tuple containing the generated data and one-hot parity labels.
    """
    x = sample_binary_data(key, n_samples, dim)
    y = vmap(one_hot_parity, in_axes=(0, None))(x, idx_mask)
    return x, y


def indices_to_mask(indices: Array, output_len) -> Bool[Array, "output_len"]:
    return jnp.zeros(output_len, dtype=bool).at[indices].set(True)


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


# TODO: This construction makes it hard to sample more data from the same task structure
# as both the task_bit_mask_array and the samples are generated from a single random key.
# Options:
#  - Have `sample_multitask_parity_data` take a task_bit_mask_array as an argument
#  - Pass two differents keys to `sample_multitask_parity_data` one for the mask and one of the samples
def sample_multitask_parity_data(
    key,
    n_samples: int,
    n_tasks: int,
    n_bits_per_task: int,
    data_dim: int,
    alpha=None,
    reuse_bits=False,
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
        key, n_tasks, n_bits_per_task, data_dim, reuse_bits=reuse_bits
    )

    key1, key2 = jr.split(key)
    task_bits = sample_task_bits(key1, n_samples, n_tasks, alpha=alpha)
    data_bits = sample_binary_data(key2, n_samples, data_dim)

    _task_parity = lambda task_idx, data: one_hot_parity(
        data, task_bit_mask_array[task_idx]
    )
    y = vmap(_task_parity, in_axes=(0, 0))(task_bits.argmax(1), data_bits)

    x = jnp.concatenate((task_bits, data_bits), axis=1)
    return x, y, task_bit_mask_array
