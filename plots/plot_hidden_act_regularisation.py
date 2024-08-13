import itertools
from jax import numpy as jnp
from jax import random as jr
from jax import Array, grad, jit, vmap
from jax.nn import relu
from jax.tree_util import tree_leaves
from matplotlib import pyplot as plt
import optax
from optax.losses import softmax_cross_entropy
import os

from boolean_circuits.models import MLP
from boolean_circuits.parity_data import one_hot_parity 
from boolean_circuits.utils.data import create_minibatches
from boolean_circuits.utils.plotting import plot_activation_for_combinations

plt.style.use('thesis')

# Current file directory
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = f"{PARENT_DIR}/figures/sparse_parity_hidden_act_regularisation.png"


def complete_parity_data(key, data_dim, k=3):
    """Calculate labelled parity data for all possible bit strings length `data_dim`."""

    subkey1, subkey2 = jr.split(key)

    x = jnp.array(list(itertools.product([0, 1], repeat=data_dim)))
    x = jr.permutation(subkey1, x, axis=0)

    idxs = jr.choice(subkey2, jnp.arange(data_dim), shape=(k,), replace=False)
    idx_mask = jnp.zeros(data_dim).at[idxs].set(True)

    y = vmap(one_hot_parity, in_axes=(0, None))(x, idx_mask)

    return x, y, idxs


def train_model_with_loss(key, loss_fn, data):
    """Train model with specified loss function."""
    x_train, y_train = data
    data_dim = x_train.shape[1]

    key, subkey = jr.split(key)
    params = model.init(subkey, jnp.ones(data_dim))
    batch_size = 64
    num_epochs = 20000

    @jit
    def update(params, x, y, opt_state):
        grads = grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state


    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    for _ in range(num_epochs):
        key, subkey = jr.split(key)
        for x_batch, y_batch in create_minibatches((x_train, y_train), batch_size, subkey):
            params, opt_state = update(params, x_batch, y_batch, opt_state)

    return params


def calculate_weighted_h_activations(params, idx_mask, data_bit_combs):
    """
    Calculate weighted hidden unit activations for all bit combinations.

    Hidden unit activations are weighted by their influence on the parity output.

    Args:
        params: model parameters.
        idx_mask: mask for the data bits within the model input.
        data_bit_combs: combination of data bits to calculate hidden activations for.

    Returns:
        weighted_h: weighted hidden unit activations
    """

    def hidden_activations(params, x):
        W_in = params["params"]["Dense_0"]["kernel"]
        bias = params["params"]["Dense_0"]["bias"]
        return relu(jnp.dot(x, W_in) + bias)

    def sample_bit_pattern(bits, idx_mask, key=None):
        if key is None:
            background = jnp.zeros(len(idx_mask))
        else:
            background = jr.bernoulli(key, 0.5, shape=idx_mask.shape)
        return background.at[idx_mask].set(bits)

    sample_data_bits = vmap(sample_bit_pattern, (0, None))(data_bit_combs, idx_mask.astype(bool))

    h = vmap(hidden_activations, (None, 0))(params, sample_data_bits)
    W_out = params['params']["Dense_1"]["kernel"]
    weighted_h = h * W_out[:,1]
    return weighted_h



if __name__ == "__main__":
    data_key = jr.PRNGKey(0)
    model = MLP(features=[32, 2])
    data_dim = 8
    x, y, idxs = complete_parity_data(data_key, data_dim)

    train_N = int(0.8 * len(x))
    x_train, y_train = x[:train_N], y[:train_N]
    x_test, y_test = x[train_N:], y[train_N:]
    train_data = (x_train, y_train)

    def l1_loss_fn(params, x, y, weight_decay=1e-3):
        """Cross entropy loss with L1 weight decay."""
        logits: Array = model.apply(params, x)  # type: ignore
        ce_loss = softmax_cross_entropy(logits, y).mean()
        
        # L1 weight decay
        l1_loss = sum(jnp.sum(jnp.abs(p)) for p in tree_leaves(params))
        
        total_loss = ce_loss + 0.5 * weight_decay * l1_loss
        return total_loss

    def l2_loss_fn(params, x, y, weight_decay=1e-3):
        """Cross entropy loss with L2 weight decay."""
        logits: Array = model.apply(params, x) # type: ignore
        ce_loss = softmax_cross_entropy(logits, y).mean()
        
        # L2 weight decay
        l2_loss = sum(jnp.sum(jnp.square(p)) for p in tree_leaves(params))
        
        total_loss = ce_loss + 0.5 * weight_decay * l2_loss
        return total_loss

    def unreg_loss_fn(params, x, y, weight_decay=1e-3):
        """Cross entropy loss."""
        logits: Array = model.apply(params, x) # type: ignore
        ce_loss = softmax_cross_entropy(logits, y).mean()
        
        total_loss = ce_loss + 0.5 * weight_decay 
        return total_loss


    l1_params = train_model_with_loss(jr.PRNGKey(101), l1_loss_fn, train_data)
    l2_params = train_model_with_loss(jr.PRNGKey(101), l2_loss_fn, train_data)
    unreg_params = train_model_with_loss(jr.PRNGKey(101), unreg_loss_fn, train_data)

    idx_mask = jnp.zeros(data_dim).at[idxs].set(True)

    bit_combs = jnp.array(list(itertools.product([0, 1], repeat=len(idxs))))
    h_unreg = calculate_weighted_h_activations(unreg_params, idx_mask, bit_combs)
    h_l2 = calculate_weighted_h_activations(l2_params, idx_mask, bit_combs)
    h_l1 = calculate_weighted_h_activations(l1_params, idx_mask, bit_combs)

    # squeeze=False needed to stop type checker complaining
    fig, axs = plt.subplots(3,1, figsize=(9, 6), squeeze=False) 
    axs = axs.squeeze()
    im1 = plot_activation_for_combinations(h_unreg, bit_combs, ax=axs[0])
    axs[0].set_title("Unregularized")
    im2 = plot_activation_for_combinations(h_l2, bit_combs, ax=axs[1])
    axs[1].set_title("L2 Regularized")
    im3 = plot_activation_for_combinations(h_l1, bit_combs, ax=axs[2])
    axs[2].set_title("L1 Regularized")

    plt.tight_layout(rect=(0, 0, 0.95, 1))

    # Add colorbar
    cbar_ax = fig.add_axes((0.88, 0.04, 0.02, 0.88))  # [left, bottom, width, height]
    cbar = fig.colorbar(im3, cax=cbar_ax, ticks=[]) 
    cbar.ax.text(0.5, 1.02, 'Parity 1', ha='center', va='bottom', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, 'Parity 0', ha='center', va='top', transform=cbar.ax.transAxes)
    # cbar.set_label('Hidden Unit to Parity Influence')

    plt.savefig(OUT_FILE,bbox_inches="tight", dpi=300)

