"""Train a simple MLP on the multi-task sparse parity problem."""
from datetime import datetime
from flax.training import checkpoints
from jax import numpy as jnp
from jax import random as jr
from jax import Array, grad, jit
import numpy as np
import optax
from optax.losses import softmax_cross_entropy

from boolean_circuits.jax.models import MLP
from boolean_circuits.jax.data.parity_data import sample_multitask_parity_data
from boolean_circuits.jax.utils import create_minibatches

key = jr.PRNGKey(0)
date = datetime.today().strftime("%Y-%m-%d_%H%M")
CHECKPOINT_ROOT = f"/Users/ghd/dev/SPAR/boolean-circuits/checkpoints/msp/{date}/"

# Task specs from Michaud2023
# - ntasks = 500
# - n = 100 (task bits)
# - k = 3 (parity bits per task)
# - α = 1.4 (NB. in paper it is parameterized as k^-(1+α))
# - batch size of 20000
# - training dataset size 1e4-5e6
# - single hidden-layer width 10-500 neurons
# - train for 2e5 steps

N = 500_000
data_bits = 100
n_tasks = 500
data_dim = data_bits + n_tasks
n_bits_per_task = 3
alpha = 1.4
key, subkey = jr.split(key)
x, y, _ = sample_multitask_parity_data(
    subkey,
    N,
    n_tasks,
    n_bits_per_task,
    data_bits,
    alpha=alpha,
    reuse_bits=True,
)

train_N = int(0.8 * N)
x_train, y_train = x[:train_N], y[:train_N]
x_test, y_test = x[train_N:], y[train_N:]

## Model ##
key, subkey = jr.split(key)
# model = MLP(features=[32, 32, 2])
model = MLP(features=[300, 2])
params = model.init(subkey, jnp.ones(data_dim))


## Training Setup ##
@jit
def loss_fn(params, x, y):
    """Cross entropy loss."""
    logits: Array = model.apply(params, x)  # type: ignore
    return softmax_cross_entropy(logits, y).mean()

# lr = 1e-3 big improvement over 1e-2
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)


@jit
def update(params, x, y, opt_state):
    grads = grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


## Training Loop ##
@jit
def accuracy(params, x, y):
    """Calculate the accuracy of `model` and `params` on a given dataset."""
    logits: Array = model.apply(params, x)  # type: ignore
    return jnp.sum(jnp.argmax(logits, axis=1) == jnp.argmax(y, axis=1)) / y.shape[0]


train_loss = []
test_loss = []

# Training loop
batch_size = 20000
num_epochs = 500
key = jr.PRNGKey(0)
training_params = {"batch_size": batch_size, "num_epochs": num_epochs, "key": key}
params_checkpoint_dir = f"{CHECKPOINT_ROOT}/params_opt"
for epoch in range(num_epochs):
    key, subkey = jr.split(key)

    for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size, subkey):
        params, opt_state = update(params, x_batch, y_batch, opt_state)

    if epoch % 2 == 0:
        checkpoint_dict = {"params": params, "opt_state": opt_state}
        checkpoints.save_checkpoint(
            params_checkpoint_dir,
            checkpoint_dict,
            prefix="epoch_",
            step=epoch,
            keep_every_n_steps=1,
        )

    current_train_loss = loss_fn(params, x_train, y_train)
    current_test_loss = loss_fn(params, x_test, y_test)
    train_loss.append(current_train_loss)
    test_loss.append(current_test_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print(f"\tTrain Loss: {current_train_loss:.4e}")
        print(f"\tTest Loss: {current_test_loss:.4e}")
        print(f"\tTrain Accuracy: {accuracy(params, x_train, y_train):.3f}")
        print(f"\tTest Accuracy: {accuracy(params, x_test, y_test):.3f}")


np.savetxt(f"{CHECKPOINT_ROOT}/train_loss.txt", jnp.array(train_loss))
np.savetxt(f"{CHECKPOINT_ROOT}/test_loss.txt", jnp.array(test_loss))

