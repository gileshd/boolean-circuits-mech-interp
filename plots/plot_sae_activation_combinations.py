import itertools
from jax import numpy as jnp
from jax import random as jr
from jax import grad, jit, vmap
from jax.nn import relu
from jax.tree_util import tree_leaves
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import optax
from optax.losses import softmax_cross_entropy
import os

from boolean_circuits.jax.models import MLP
from boolean_circuits.jax.data.parity_data import sample_binary_parity_data
from boolean_circuits.jax.sae import AutoEncoder
from boolean_circuits.jax.utils.data import create_minibatches
from boolean_circuits.jax.utils.plotting import plot_activation_for_combinations

plt.style.use("thesis")

# Current file directory
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = f"{PARENT_DIR}/figures/sparse_parity_sae_act.png"


def train_model(key, data, num_epochs=10000, batch_size=64):

    x_train, y_train, *_ = data
    data_dim = x_train.shape[1]
    model = MLP(features=[32, 2])
    key = jr.PRNGKey(0)
    params = model.init(key, jnp.ones(data_dim))

    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    def loss_l2(params, x, y, weight_decay=1e-3):
        """Cross entropy loss with L2 weight decay."""
        logits = model.apply(params, x)
        ce_loss = softmax_cross_entropy(logits, y).mean()  # type: ignore
        l2_loss = sum(jnp.sum(jnp.square(p)) for p in tree_leaves(params))

        total_loss = ce_loss + 0.5 * weight_decay * l2_loss
        return total_loss

    @jit
    def update(params, x, y, opt_state):
        grads = grad(loss_l2)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    params = model.init(key, jnp.ones(data_dim))
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    for _ in range(num_epochs):
        key, subkey = jr.split(key)
        for x_batch, y_batch in create_minibatches(
            (x_train, y_train), batch_size, subkey
        ):
            params, opt_state = update(params, x_batch, y_batch, opt_state)

    return model, params


def train_sae(key, model_params, data, num_epochs=10000, batch_size=64):

    ## SAE ##
    sae_input_dim = model_params["params"]["Dense_0"]["kernel"].shape[1]
    sae_latent_dim = 32

    sae = AutoEncoder(input_dim=sae_input_dim, hidden_dim=sae_latent_dim)

    ## Data ##
    @jit
    def hidden_activations(params, x):
        W_in = params["params"]["Dense_0"]["kernel"]
        bias = params["params"]["Dense_0"]["bias"]
        return relu(jnp.dot(x, W_in) + bias)

    h_act = vmap(hidden_activations, (None, 0))(model_params, data)

    ## Training ##
    def loss_sae(params, x, lam=0.1):
        """Sum sq loss."""
        x_recon = sae.apply(params, x)
        hidden = sae.apply(params, x, method=sae.encode)
        return jnp.mean((x - x_recon) ** 2) + lam * jnp.abs(hidden).mean()  # type: ignore

    optimizer = optax.adam(learning_rate=0.01)

    @jit
    def update(params, x, opt_state):
        grads = grad(loss_sae)(params, x)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    key, subkey = jr.split(key)
    sae_params = sae.init(subkey, jnp.ones(sae_input_dim))
    opt_state = optimizer.init(sae_params)

    for _ in range(num_epochs):
        key, subkey = jr.split(key)
        for x_batch, *_ in create_minibatches((h_act,), batch_size, subkey):
            sae_params, opt_state = update(sae_params, x_batch, opt_state)

    return sae, sae_params


def calculate_weighted_sae_activations(
    model_params, sae, sae_params, data_bit_combs, idx_mask
):
    """Calculate weighted activations of sae hidden units for all bit combinations in `data_bit_combs`."""

    def hidden_activations(params, x):
        W_in = params["params"]["Dense_0"]["kernel"]
        bias = params["params"]["Dense_0"]["bias"]
        return relu(jnp.dot(x, W_in) + bias)

    @jit
    def sae_activations(sae_params, model_params, x):
        h = hidden_activations(model_params, x)
        return sae.apply(sae_params, h, method=sae.encode)

    def sample_bit_pattern(bits, idx_mask, key=None):
        """
        Sample a bit pattern with `bits` set at `idx_mask`.

        If key is None return zeros in all non-masked positions, else sample from bernoulli.
        """
        if key is None:
            background = jnp.zeros(len(idx_mask))
        else:
            background = jr.bernoulli(key, 0.5, shape=idx_mask.shape)
        return background.at[idx_mask].set(bits)

    data_bit_combs_with_zeros = vmap(sample_bit_pattern, (0, None))(
        data_bit_combs, idx_mask.astype(bool)
    )

    s_act = sae_activations(sae_params, model_params, data_bit_combs_with_zeros)

    W_sae_decode = sae_params["params"]["decoder"]["kernel"]
    W_model_out = model_params["params"]["Dense_1"]["kernel"]
    W_sae_out = W_sae_decode @ W_model_out
    weight_sae_parity = W_sae_out @ jnp.array([-1, 1])

    weighted_sae_activations = s_act * weight_sae_parity

    return weighted_sae_activations


if __name__ == "__main__":
    N = 256
    data_dim = 8

    ## Data ##
    data_key = jr.PRNGKey(10)
    subkey1, subkey2 = jr.split(data_key)
    idxs = jr.choice(data_key, jnp.arange(data_dim), shape=(3,), replace=False)
    idx_mask = jnp.zeros(data_dim).at[idxs].set(True)
    x, y = sample_binary_parity_data(subkey2, N, data_dim, idx_mask)

    train_N = int(0.8 * N)
    x_train, y_train = x[:train_N], y[:train_N]
    x_test, y_test = x[train_N:], y[train_N:]
    data = (x_train, y_train, x_test, y_test)

    ## Train Models ##
    key = jr.PRNGKey(0)
    model, model_params = train_model(key, data)
    sae, sae_params = train_sae(key, model_params, x_train, num_epochs=10000)

    ## Calculate hidden unit activations ##
    bit_combs = jnp.array(list(itertools.product([0, 1], repeat=len(idxs))))
    weighted_sae_activations = calculate_weighted_sae_activations(
        model_params, sae, sae_params, bit_combs, idx_mask
    )

    ## Plotting ##
    fig, ax = plt.subplots(figsize=(12, 6))
    im = plot_activation_for_combinations(weighted_sae_activations, bit_combs, ax=ax)
    
    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)

    # Append an axes to the right of ax, with the width roughly proportional to the heatmap
    cax = divider.append_axes("right", size="2%", pad=0.2)

    # Create the colorbar in the new axis
    cbar = fig.colorbar(im, cax=cax, ticks=[])
    cbar.ax.text(
        0.5, 1.02, "Parity 1", ha="center", va="bottom", transform=cbar.ax.transAxes
    )
    cbar.ax.text(
        0.5, -0.02, "Parity 0", ha="center", va="top", transform=cbar.ax.transAxes
    )

    plt.tight_layout()
    plt.savefig(OUT_FILE, bbox_inches="tight", dpi=300)
