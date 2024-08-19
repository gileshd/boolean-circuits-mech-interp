from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import tree_map
from jaxtyping import Array, Int


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
    assert (
        len(set(tree_map(lambda a: a.shape[0], data))) == 1
    ), "All arrays must have the same first dimension"
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


# TODO: The type hints here should really be `Union[Bool, Int]` but 
#        I'm not sure how to get jaxtyping to play nice with this this...
def sort_binary_array(
    x: Int[Array, "n_samples n_bits"]
) -> tuple[Int[Array, "n_samples n_bits"], Int[Array, "n_samples"]]:
    """Sort rows of binary array according to their integer value.

    Args:
        x: Binary array of shape (n_samples, n_bits) where each row is a binary number.

    Returns:
        Tuple: Sorted binary array and the indices that were used to sort the array
    """

    sorted_idx = jnp.argsort(jnp.packbits(x, axis=1).flatten())
    return x[sorted_idx], sorted_idx
