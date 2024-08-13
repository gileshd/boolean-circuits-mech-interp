from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import tree_map

def create_minibatches(data, batch_size, rng_key):
    """
    Generate minibatches of data and labels, shuffled at each epoch.

    Parameters
    ----------
    x : PyTree with leaves of type jnp.ndarray
    batch_size : int
        Size of each minibatch.
    rng_key : jax.random.PRNGKey
        JAX random key for shuffling.

    Returns
    -------
    generator
        A generator that yields PyTrees batched data.
    """
    assert len(set(tree_map(lambda a: a.shape[0], data))) == 1, "All arrays must have the same first dimension"
    num_examples = data[0].shape[0]

    # Shuffle data and labels in unison
    indices = jnp.arange(num_examples)
    shuffled_indices = jr.permutation(rng_key, indices)

    shuffled_data = tree_map(lambda a: a[shuffled_indices], data)

    # Generate minibatches
    num_batches = num_examples // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = tree_map(lambda a: a[start_idx:end_idx], shuffled_data)
        yield batch
