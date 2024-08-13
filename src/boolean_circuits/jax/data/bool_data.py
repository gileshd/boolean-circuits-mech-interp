from jax import vmap
from jax import numpy as jnp
from jax import random as jr


def n_to_binary_array(number, num_bits=8):
    """
    Convert a number to its binary representation as a JAX array.
    
    Args:
    number (int): The number to convert.
    num_bits (int): The number of bits to use for the representation.
    
    Returns:
    jax.Array: A JAX array containing the binary representation.
    """
    binary = jnp.arange(num_bits - 1,-1,-1)
    number = jnp.where(number < 2**num_bits, number, jnp.nan)
    return (number // (2 ** binary)) % 2

def binary_array_to_number(binary_array):
    """
    Convert a binary representation (JAX array) back to its decimal number.
    The input array should have the least significant bit as the rightmost element.
    
    Args:
    binary_array (jax.Array): The binary representation as a JAX array.
    
    Returns:
    int: The decimal representation of the binary array.
    """
    num_bits = binary_array.shape[0]
    powers = jnp.arange(num_bits - 1, -1, -1)
    return jnp.sum(binary_array * (2 ** powers))

def generate_bin_addition_data(key, n_samples=100):
    """
    Generate data for the binary addition task.
    """
    min_val = 0
    max_val = 2**8 - 1
    subkey1, subkey2 = jr.split(key)
    a = jr.randint(subkey1, (n_samples,), min_val, max_val)
    b = jr.randint(subkey2, (n_samples,), min_val, max_val)
    c = a + b

    a_bin = vmap(n_to_binary_array)(a)
    b_bin = vmap(n_to_binary_array)(b)
    x = jnp.concatenate([a_bin, b_bin], axis=1) 
    y = vmap(lambda n: n_to_binary_array(n, num_bits=9))(c)
    return x,y
