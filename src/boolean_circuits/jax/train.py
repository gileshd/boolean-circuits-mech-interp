import jax
from jax import grad, jit
from jax import numpy as jnp
from jax import random as jr
from flax import linen as nn
import optax
from optax.losses import softmax_cross_entropy

from boolean_circuits.jax.utils.data import create_minibatches, actualise_minibatches


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
    model: nn.Module,
    num_epochs=1000,
    batch_size=64,
    learning_rate=0.001,
    print_every=None,
):
    """
    Train model paramters.
    """

    x, y = data
    optimizer = optax.adam(learning_rate=learning_rate)

    @jit
    def update(params, x, y, opt_state):
        grads = grad(loss_fn)(params, x, y, model)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    params = model.init(key, jnp.zeros_like(x[0]))
    opt_state = optimizer.init(params)
    for epoch in range(num_epochs):
        key, subkey = jr.split(key)
        for x_batch, y_batch in create_minibatches((x, y), batch_size, subkey):
            params, opt_state = update(params, x_batch, y_batch, opt_state)
        if print_every is not None and epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {loss_fn(params, x, y, model)}")

    return params


def train_MLP_scan(
    key,
    data,
    loss_fn,
    model: nn.Module,
    num_epochs=1000,
    batch_size=64,
    learning_rate=0.001,
    save_every=100,
):
    """
    Train parameters of `model` return final and intermediate parameters.
     
    Uses nested `jax.lax.scan` to accumulate parameters evey `save_every` epoch into leading axis
    of `saved_params`.

    Returns a tuple of the final parameter PyTree and the PyTree of saved parameters. Saved 
    parameters include the final parameters as last entry.
    """
    x, y = data
    optimizer = optax.adam(learning_rate=learning_rate)

    def update_step(carry, batch):
        """Update step with signature to work with jax.lax.scan."""
        params, opt_state = carry
        x_batch, y_batch = batch
        grads = grad(loss_fn)(params, x_batch, y_batch, model)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), None

    def train_single_epoch(carry, _):
        """Train for a single epoch."""
        params, opt_state, key = carry
        key, subkey = jr.split(key)
        batches = actualise_minibatches((x, y), batch_size, subkey)
        (params, opt_state), _ = jax.lax.scan(update_step, (params, opt_state), batches)
        return (params, opt_state, key), None

    @jit
    def scan_multi_epochs(carry, _):
        """Train for `save_every` epochs."""
        params, opt_state, key = carry
        (params, opt_state, key), _ = jax.lax.scan(
            train_single_epoch, (params, opt_state, key), None, length=save_every
        )
        return (params, opt_state, key), params

    init_params = model.init(key, jnp.zeros_like(x[0]))
    init_opt_state = optimizer.init(init_params)

    num_outer_steps = num_epochs // save_every

    # Scan each step in scan trains for `save_every` epochs and accumulates the values in
    # the leading axis of saved_params.
    (final_params, _, _), saved_params = jax.lax.scan(
        scan_multi_epochs,
        (init_params, init_opt_state, key),
        None,
        length=num_outer_steps,
    )

    return final_params, saved_params
