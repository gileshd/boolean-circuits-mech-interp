import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_offset_vlines(x, ys, offset_range=0.2, ax=None):
    """Plot vertical lines with small x offsets."""
    if ax is None:
        _, ax = plt.subplots()

    offsets = np.linspace(-offset_range/2, offset_range/2, len(ys))
    for n, (y, off) in enumerate(zip(ys, offsets)):
        xs = x + off
        ax.plot(xs, y, '.', color=f'C{n}');
        ax.vlines(xs, 0, y, color=f'C{n}', alpha=0.5);
    return ax


def plot_SP_weights(params, data_bit_idxs, axs=None):
    """
    Visualisation of weights and biases from input to hidden layer.
 
    Plot weights from input to hidden layer for each of the data bits in the input alongside biases.
    Separate hidden units according to their influence of the model output.

    Args:
        params: model parameters
        data_bit_idxs: indices of the data bits in the input
        axs (optional): list of axes objects for the plots
    Returns:
        axs: list of axes objects for the plots
    """

    # Extract model parameters
    W_in = params["params"]["Dense_0"]["kernel"]
    W_in = W_in[data_bit_idxs] # only plot the weights for the data bits
    bias_in = params["params"]["Dense_0"]["bias"]
    W_out = params['params']["Dense_1"]["kernel"]

    # Calculate parity masks
    parity_measure = W_out @ np.array([-1,1])
    par0_mask = parity_measure < 0
    par1_mask = parity_measure > 0
    masks = (par0_mask, par1_mask)

    if axs is None:
        _, axs = plt.subplots(1,2, figsize=(10,5), squeeze=False)
        axs = axs.squeeze() # type checkers are silly and they make me sad

    titles = ("Parity 0", "Parity 1")
    units = np.arange(W_in.shape[1])
    for ax, mask, title in zip(axs, masks, titles):
        plot_offset_vlines(units[mask], W_in[:,mask], offset_range=0.3, ax=ax)
        ax.plot(units[mask],bias_in[mask], "*", color="grey", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Hidden Units")

    axs[0].set_ylabel("Weights from input to hidden layer")
    return axs


def plot_activation_for_combinations(x, bit_combs, ax=None, color_yticks=True):
    """
    Heatmap of activations in response to inputs `bit_combs` in input.

    Args:
        x: Array of activations.
        bit_combs: Array of bit combinations.
        ax (optional): Matplotlib axis object.
        color_yticks (optional): Boolean to color y-ticks according to parity of 
                                 corresponding bit combination.
    """

    if ax is None:
        _, ax = plt.subplots()

    vmin, vmax = x.min(), x.max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(x, cmap='RdBu', norm=norm)
    ax.set_yticks(range(8), labels= [str(r) for r in bit_combs]);
    ax.set_xticks([])

    if color_yticks:
        cmap = plt.get_cmap('RdBu')
        colors = [cmap.get_over(), cmap.get_under()]
        par = lambda x: int(sum(x) % 2 == 0)
        ytick_colors = [colors[par(r)] for r in bit_combs]

        for label, color in zip(ax.get_yticklabels(), ytick_colors):
            label.set_color(color) # type: ignore

    return im


def plot_circuit(circuit, ax=None) -> None:
    G = nx.DiGraph()
    pos = {}
    labels = {}

    # Assign unique IDs to input nodes
    for i in range(circuit.input_size):
        node_id = f"input_{i}"
        G.add_node(node_id)
        pos[node_id] = (-1, -i)
        labels[node_id] = f"Input {i}"

    # Assign unique IDs to all gates
    for layer_idx, layer in enumerate(circuit.layers):
        for gate_idx, gate in enumerate(layer.gates):
            node_id = f"layer_{layer_idx}_gate_{gate_idx}"
            G.add_node(node_id)
            pos[node_id] = (layer_idx, -gate_idx)
            labels[node_id] = str(gate.operation)

            # Add edges based on gate inputs
            for input_idx in gate.input_idxs:
                if layer_idx == 0:
                    G.add_edge(f"input_{input_idx}", node_id)
                else:
                    prev_layer_len = len(circuit.layers[layer_idx - 1])
                    for prev_gate_idx in range(prev_layer_len):
                        if prev_gate_idx == input_idx:
                            G.add_edge(
                                f"layer_{layer_idx-1}_gate_{prev_gate_idx}", node_id
                            )
                            break

    # Add output node
    output_node = "output"
    G.add_node(output_node)
    pos[output_node] = (len(circuit.layers), 0)
    labels[output_node] = str(circuit.output_gate.operation)

    # Connect output gate to output node
    for gate_idx in circuit.output_gate.input_idxs:
        G.add_edge(f"layer_{len(circuit.layers)-1}_gate_{gate_idx}", output_node)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        ax=ax,
    )
    plt.title("Boolean Circuit Computational Graph")
    plt.show()
