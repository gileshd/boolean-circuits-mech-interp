import jax
from jax import jit, grad
from jax import numpy as jnp
from jax import random as jr
import optax
from optax.losses import softmax_cross_entropy

from boolean_circuits.jax.data.parity_data import sample_binary_parity_data
from boolean_circuits.jax.models import MLPWithIntermediates
from boolean_circuits.jax.utils.data import create_minibatches
from boolean_circuits.utils import all_combinations


def sample_sparse_parity_data(key, n_samples: int, n_data_bits: int, dim: int):
    idx_key, data_key = jr.split(key)
    idxs = jr.choice(idx_key, jnp.arange(dim), shape=(n_data_bits,), replace=False)
    idx_mask = jnp.zeros(dim).at[idxs].set(True)
    x, y = sample_binary_parity_data(data_key, n_samples, dim, idx_mask)
    return idxs, x, y


def train_MLP(key, data, num_epochs=1000, batch_size = 64, learning_rate=0.001, print_every=None):

    x, y = data
    model = MLPWithIntermediates(features=[32,2])
    optimizer = optax.adam(learning_rate=learning_rate)

    def loss_fn(params, x, y, weight_decay=1e-3):
        """Cross entropy loss with L2 weight decay."""
        logits, *_  = model.apply(params, x)
        ce_loss = softmax_cross_entropy(logits, y).mean() 
        l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))

        total_loss = ce_loss + 0.5 * weight_decay * l2_loss
        return total_loss
    
    @jit
    def update(params, x, y, opt_state):
        grads = grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    data_dim = x.shape[-1]
    params = model.init(key, jnp.ones(data_dim))
    opt_state = optimizer.init(params)
    for epoch in range(num_epochs):
        key, subkey = jr.split(key)
        for x_batch, y_batch in create_minibatches((x, y), batch_size, subkey):
            params, opt_state = update(params, x_batch, y_batch, opt_state)
        if print_every is not None and epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {loss_fn(params, x, y)}")

    return model, params

def accuracy(model, params, x, y):
    logits, *_ = model.apply(params, x)
    return jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))

data_dim = 8
N = 256
data_key = jr.PRNGKey(10)
idxs, x, y = sample_sparse_parity_data(data_key, N, 4, data_dim)

train_key = jr.PRNGKey(20)
print("### Training MLP: ###")
model, params = train_MLP(train_key, (x, y), num_epochs=6000, batch_size=64, learning_rate=0.001, print_every=1000)
print(f"Final Accuracy: {accuracy(model, params, x, y)}")

# Train probes
idx_combs = all_combinations(idxs, min_length=1)

