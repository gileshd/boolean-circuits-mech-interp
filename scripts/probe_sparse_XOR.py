"""
Train probes to predict parity of subsets of data bits from hidden activations 
of MLP trained on sparse parity problem.
"""

import jax
from jax import grad, jit, vmap
from jax import numpy as jnp
from jax import random as jr
import optax
from optax.losses import softmax_cross_entropy

from boolean_circuits.jax.data.parity_data import parity, sample_binary_parity_data
from boolean_circuits.jax.probes import linear, train_logistic_probe
from boolean_circuits.jax.models import MLPWithIntermediates
from boolean_circuits.jax.utils.data import create_minibatches
from boolean_circuits.utils import all_combinations


def sample_sparse_parity_data(key, n_samples: int, n_data_bits: int, dim: int):
    """
    Sample sparse parity data.

    Args:
        key (jax.random.PRNGKey): Random key.
        n_samples (int): Number of samples to generate.
        n_data_bits (int): Number of data bits.
        dim (int): Dimension of the data.

    Returns:
        idxs (Array - (n_data_bits,)): indices of data bits within x
        x (Array - (n_samples, dim)): binary data
        y (Array - (n_samples,): parity of data bits within x
    """
    idx_key, data_key = jr.split(key)
    idxs = jr.choice(idx_key, jnp.arange(dim), shape=(n_data_bits,), replace=False)
    idx_mask = jnp.zeros(dim).at[idxs].set(True)
    x, y = sample_binary_parity_data(data_key, n_samples, dim, idx_mask)
    return idxs, x, y


def loss_l2(params, x, y, model, weight_decay=1e-3):
    """Cross entropy loss with L2 weight decay."""
    logits, *_ = model.apply(params, x)
    ce_loss = softmax_cross_entropy(logits, y).mean()
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))

    total_loss = ce_loss + 0.5 * weight_decay * l2_loss
    return total_loss


def loss_l1(params, x, y, model, weight_decay=1e-3):
    """Cross entropy loss with L1 weight decay."""
    logits, *_ = model.apply(params, x)
    ce_loss = softmax_cross_entropy(logits, y).mean()
    l1_loss = sum(jnp.sum(jnp.abs(p)) for p in jax.tree.leaves(params))

    total_loss = ce_loss + 0.5 * weight_decay * l1_loss
    return total_loss


def train_MLP(
    key,
    data,
    loss_fn,
    num_epochs=1000,
    batch_size=64,
    learning_rate=0.001,
    print_every=None,
):

    x, y = data
    model = MLPWithIntermediates(features=[32, 2])
    optimizer = optax.adam(learning_rate=learning_rate)

    @jit
    def update(params, x, y, opt_state):
        grads = grad(loss_fn)(params, x, y, model)
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
            print(f"Epoch {epoch}, Loss: {loss_fn(params, x, y, model)}")

    return model, params


def probe_for_idx_parities(key, idx_mask, model, params, x):
    """
    Probe hidden activations for the parities of the data bits at the given indices.
    """

    # Calculate hidden activations
    _, activations = model.apply(params, x)
    h = activations["layer_0"]

    # Calculate parity of idxs for each sample
    parities = vmap(parity, (0, None))(x, idx_mask)

    probe_params, probe_loss = train_logistic_probe(key, h, parities)
    probe_acc = probe_accuracy(probe_params, h, parities)

    return probe_params, probe_loss, probe_acc


def model_accuracy(model, params, x, y):
    logits, *_ = model.apply(params, x)
    return jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))


def probe_accuracy(probe_params, activations, target_labels):
    logits = vmap(linear, (None, 0))(probe_params, activations)
    return jnp.mean((logits > 0) == target_labels)


if __name__ == "__main__":
    ## Generate Data ##
    data_dim = 8
    n_data_bits = 4
    N = 256
    data_key = jr.PRNGKey(10)
    idxs, x, y = sample_sparse_parity_data(data_key, N, n_data_bits, data_dim)
    print("### Data: ###")
    print("Sparse Parity Data:")
    print(f"\tData dim: {data_dim}, Number of data bits: {n_data_bits}")
    print()

    ## Train Model ##
    train_key = jr.PRNGKey(20)
    print("### Training MLPs: ###")

    print("Train MLP with L2 regularization:")
    N_epochs = 6000
    model, params_l2 = train_MLP(
        train_key,
        (x, y),
        loss_l2,
        num_epochs=N_epochs,
        batch_size=64,
        learning_rate=0.001,
        print_every=1000,
    )
    print(f"L2 Final Accuracy: {model_accuracy(model, params_l2, x, y)}")
    print()

    print("Train MLP with L1 regularization:")
    N_epochs = 6000
    model, params_l1 = train_MLP(
        train_key,
        (x, y),
        loss_l1,
        num_epochs=N_epochs,
        batch_size=64,
        learning_rate=0.001,
        print_every=1000,
    )
    print(f"Final Accuracy: {model_accuracy(model, params_l1, x, y)}")
    print()

    ## Train probes ##
    print("### Training Probes: ###")
    print("Train probes for parities of all combinations of data bits")

    # Generate all subcombinations of data bits
    idx_combs = list(jnp.array(comb) for comb in all_combinations(idxs, min_length=1))
    idx_to_mask = lambda idxs: jnp.zeros(data_dim).at[idxs].set(True)
    idx_combs_masks = jnp.array([idx_to_mask(i) for i in idx_combs])

    probe_keys = jr.split(jr.PRNGKey(30), len(idx_combs))

    print()
    print("Probe Accuracies on L2 MLP:")
    # Map probe train function over all combinations of data bits
    *_, probes_acc_l2 = vmap(probe_for_idx_parities, (0, 0, None, None, None))(
        probe_keys, idx_combs_masks, model, params_l2, x
    )
    for ic, acc in zip(idx_combs, probes_acc_l2):
        print(f"Indices: {ic}, Probe Accuracy: {acc}")

    print()
    print("Probe Accuracies on L1 MLP:")
    # Map probe train function over all combinations of data bits
    *_, probes_acc_l1 = vmap(probe_for_idx_parities, (0, 0, None, None, None))(
        probe_keys, idx_combs_masks, model, params_l1, x
    )
    for ic, acc in zip(idx_combs, probes_acc_l1):
        print(f"Indices: {ic}, Probe Accuracy: {acc}")
