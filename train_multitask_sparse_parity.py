from jax import numpy as jnp
from jax import random as jr
from jax import Array, grad, jit
import optax
from optax.losses import softmax_cross_entropy

from model import MLP
from parity_data import sample_multitask_parity_data
from utils import create_minibatches

## Model ##
data_bits = 16
n_tasks = 3
data_dim = data_bits + n_tasks

model = MLP(features=[32, 32, 2])
key = jr.PRNGKey(0)
params = model.init(key, jnp.ones(data_dim))

## Data ##
N = 5000
key, subkey = jr.split(key)
n_bits_per_task = 3
x, y, _ = sample_multitask_parity_data(subkey, N, n_tasks, n_bits_per_task, data_bits)

train_N = int(0.8 * N)
x_train, y_train = x[:train_N], y[:train_N]
x_test, y_test = x[train_N:], y[train_N:]

## Training Setup ##
def loss_fn(params, x, y):
    """Cross entropy loss."""
    logits: Array = model.apply(params, x) # type: ignore
    return softmax_cross_entropy(logits, y).mean()


optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

@jit
def update(params, x, y, opt_state):
    grads = grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


## Training Loop ##
def accuracy(params, x, y):
    """Calculate the accuracy of `model` and `params` on a given dataset."""
    logits: Array = model.apply(params, x)  # type: ignore
    return sum(jnp.argmax(logits, axis=1) == jnp.argmax(y, axis=1)) / y.shape[0]


train_loss = []
test_loss = []

# Training loop
batch_size = 128
num_epochs = 100
key = jr.PRNGKey(0)
for epoch in range(num_epochs):
    key, subkey = jr.split(key)

    for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size, subkey):
        params, opt_state = update(params, x_batch, y_batch, opt_state)

    if epoch % 10 == 0:
        current_train_loss = loss_fn(params, x_train, y_train)
        current_test_loss = loss_fn(params, x_test, y_test)
        train_loss.append(current_train_loss)
        test_loss.append(current_test_loss)
        print(f"Epoch {epoch}")
        print(f"\tTrain Loss: {current_train_loss:.4e}")
        print(f"\tTest Loss: {current_test_loss:.4e}")
        print(f"\tTrain Accuracy: {accuracy(params, x_train, y_train):.3f}")
        print(f"\tTest Accuracy: {accuracy(params, x_test, y_test):.3f}")
