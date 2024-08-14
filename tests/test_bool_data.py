from jax import numpy as jnp

from boolean_circuits.jax.data.bool_data import n_to_binary_array, binary_array_to_number

def assert_all_equal(x, y):
    assert jnp.all(x == y)

def test_n_to_bin_array():
    # Test values
    assert_all_equal(n_to_binary_array(0), jnp.zeros(8))
    assert_all_equal(n_to_binary_array(1), jnp.array([0, 0, 0, 0, 0, 0, 0, 1]))
    assert_all_equal(n_to_binary_array(10), jnp.array([0, 0, 0, 0, 1, 0, 1, 0]))
    assert_all_equal(n_to_binary_array(2**8 - 1), jnp.ones(8))

    # Test shapes
    assert_all_equal(n_to_binary_array(0, num_bits=4), jnp.zeros(4))

    # Test input out of bounds
    nan_output = n_to_binary_array(256)
    assert nan_output.shape == (8,)
    assert all(jnp.isnan(nan_output))


def test_bin_array_to_number():
    assert binary_array_to_number(jnp.zeros(8)) == 0
    assert binary_array_to_number(jnp.array([0, 0, 0, 0, 0, 0, 0, 1])) == 1
    assert binary_array_to_number(jnp.array([0, 0, 0, 0, 1, 0, 1, 0])) == 10
    assert binary_array_to_number(jnp.ones(8)) == 2**8 -1
