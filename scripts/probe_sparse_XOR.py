"""
Train probes to predict parity of subsets of data bits from hidden activations 
of MLP trained on sparse parity problem.
"""

from jax import vmap
from jax import numpy as jnp
from jax import random as jr

from boolean_circuits.jax.data.parity_data import parity, sample_binary_parity_data
from boolean_circuits.jax.models import MLPWithIntermediates
from boolean_circuits.jax.probes import linear, train_logistic_probe
from boolean_circuits.jax.train import loss_l1, loss_l2, train_MLP_scan


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

    from boolean_circuits.utils import all_combinations

    model = MLPWithIntermediates(features=[32, 2])

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
    params_l2, saved_params_l1 = train_MLP_scan(
        train_key,
        (x, y),
        loss_l2,
        model,
        num_epochs=N_epochs,
        batch_size=64,
        learning_rate=0.001,
    )
    print(f"L2 Final Accuracy: {model_accuracy(model, params_l2, x, y)}")
    print()

    print("Train MLP with L1 regularization:")
    N_epochs = 6000
    params_l1, saved_params_l1 = train_MLP_scan(
        train_key,
        (x, y),
        loss_l1,
        model,
        num_epochs=N_epochs,
        batch_size=64,
        learning_rate=0.001,
    )
    print(f"L1 Final Accuracy: {model_accuracy(model, params_l1, x, y)}")
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
