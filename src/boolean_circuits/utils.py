from jax import numpy as jnp
from jax import random as jr

def create_minibatches(x, y, batch_size, rng_key):
    """
    Generate minibatches of data and labels, shuffled at each epoch.

    Parameters
    ----------
    x : array_like
        Input data.
    y : array_like
        Labels.
    batch_size : int
        Size of each minibatch.
    rng_key : jax.random.PRNGKey
        JAX random key for shuffling.

    Returns
    -------
    generator
        A generator that yields tuples of (x_batch, y_batch).
    """
    num_examples = x.shape[0]

    # Shuffle data and labels in unison
    indices = jnp.arange(num_examples)
    shuffled_indices = jr.permutation(rng_key, indices)

    x_shuffled = x[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    # Generate minibatches
    num_batches = num_examples // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        x_batch = x_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        yield x_batch, y_batch


