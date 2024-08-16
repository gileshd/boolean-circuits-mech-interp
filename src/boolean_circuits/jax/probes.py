import jax
from jax import jit, value_and_grad, vmap
from jax import numpy as jnp
from jax import random as jr
import optax
from optax.losses import sigmoid_binary_cross_entropy

# TODO: Maybe this should just be a Flax module?
#       or maybe now is the time to just use equinox?

def init_linear_params(key, input_dim):
    """Initialize linear layer parameters."""
    w_key, b_key = jr.split(key)
    w = jr.normal(w_key, (input_dim,))
    b = jr.normal(b_key)
    return (w, b)


def linear(params, x):
    """Logistic regression probes."""
    w, b = params
    return jnp.dot(x, w) + b


def cross_entropy_loss(params, x, y):
    """Cross-entropy loss."""
    logits = vmap(linear, (None, 0))(params, x)
    return sigmoid_binary_cross_entropy(logits, y).mean()


# TODO: Add docs & annotations
def train_logistic_probe(
    key, activations, target_labels, num_epochs=1000, learning_rate=0.01, print_every=None
):
    """Train a logistic regression probe."""
    key, subkey = jr.split(key)
    input_dim = activations.shape[-1]

    optimizer = optax.adam(learning_rate)

    @jit
    def update(params, x, y, opt_state):
        loss, grads = value_and_grad(cross_entropy_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params = init_linear_params(subkey, input_dim)
    opt_state = optimizer.init(params)

    for epoch in range(num_epochs):
        params, opt_state, loss = update(params, activations, target_labels, opt_state)
        if print_every and epoch % print_every == 0:
            jax.debug.print("Epoch {epoch}, Loss: {loss}", epoch=epoch, loss=loss)

    return params, loss
