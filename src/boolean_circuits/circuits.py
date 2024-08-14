from abc import abstractmethod
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

Bool = np.bool_
BoolArray = NDArray[Bool]
IntArray = NDArray[np.int_]

# TOOD: Add docstrings
# TODO: Jaxify - (equinox?)


class Operation:
    @abstractmethod
    def __call__(self, input_values: BoolArray) -> Bool:
        """Perform the operation on the input values."""
        raise NotImplementedError("Subclasses should implement this method")

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()


class AND(Operation):
    def __call__(self, input_values: BoolArray) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return np.all(input_values)


class OR(Operation):
    def __call__(self, input_values: BoolArray) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return np.any(input_values)


class NOT(Operation):
    def __call__(self, input_values: BoolArray) -> Bool:
        if len(input_values) != 1:
            raise ValueError("NOT operation takes exactly one input")
        return np.bitwise_not(input_values[0])


class NAND(Operation):
    def __call__(self, input_values: BoolArray) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return np.bitwise_not(np.all(input_values))


class NOR(Operation):
    def __call__(self, input_values: BoolArray) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return np.bitwise_not(np.any(input_values))


class XOR(Operation):
    def __call__(self, input_values: BoolArray) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return np.sum(input_values) % 2 == 1


class NOOP(Operation):
    def __call__(self, input_values: BoolArray) -> BoolArray:
        return input_values


class Gate:
    def __init__(self, operation: Operation, input_idxs: NDArray[np.int_]):
        self.operation = operation
        self.input_idxs = input_idxs

    def __call__(self, input_values: BoolArray) -> Bool:
        """Call the gate operation on `input_idxs` within `input values`."""
        return self.operation(input_values[self.input_idxs])  # type: ignore

    def __str__(self) -> str:
        return f"{self.operation}({self.input_idxs})"

    def __repr__(self) -> str:
        return self.__str__()


class Layer:
    def __init__(self, gates: list[Gate]):
        self.gates = gates

    def __call__(self, input_values: BoolArray) -> BoolArray:
        """Call each gate in the layer and combine the outputs into an array."""
        return np.array([gate(input_values) for gate in self.gates], dtype=Bool)

    def __str__(self) -> str:
        return "; ".join([str(gate) for gate in self.gates])

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.gates)

    @property
    def _max_input_idx(self):
        """The maximum input index found in a gate in the layer."""
        return max(max(gate.input_idxs) for gate in self.gates)

    def _check_idxs_present(self, prev_layer):
        """Check that all gate indices are present in the `prev_layer`."""
        max_idx = self._max_input_idx
        if max_idx > len(prev_layer.gates) - 1:
            raise ValueError("Not all gate indices not present in previous layer")


class Circuit:
    def __init__(self, layers: list[Layer], output_gate: Gate, input_size=None):
        self.layers = layers
        self.output_gate = output_gate
        self.input_size = self.layers[0]._max_input_idx + 1 if input_size is None else input_size
        self._check_wiring()

    def __call__(self, input_values: BoolArray) -> tuple[BoolArray, list[BoolArray]]:
        intermediate_values = []
        for layer in self.layers:
            input_values = layer(input_values)
            intermediate_values.append(input_values.astype(int))
        output = self.output_gate(input_values)
        return np.array(output.astype(int)), intermediate_values

    def __str__(self) -> str:
        output = "\n".join([str(layer) for layer in self.layers])
        return output + f"\n{self.output_gate}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def size(self) -> int:
        """Return the number of gates in the circuit."""
        return sum(len(layer) for layer in self.layers) + 1

    def _check_wiring(self):
        """Ensure that no gate is referring to an input that doesn't exist."""
        if max(self.output_gate.input_idxs) > len(self.layers[-1]):
            raise ValueError
        # TODO: Make this not awful - add reference to layer number in error message.
        for layer, prev_layer in zip(self.layers[-1:0:-1], self.layers[-2::-1]):
            layer._check_idxs_present(prev_layer)

    def plot_circuit(self) -> None:
        G = nx.DiGraph()
        pos = {}
        labels = {}

        # Assign unique IDs to input nodes
        for i in range(self.input_size):
            node_id = f"input_{i}"
            G.add_node(node_id)
            pos[node_id] = (-1, -i)
            labels[node_id] = f"Input {i}"

        # Assign unique IDs to all gates
        for layer_idx, layer in enumerate(self.layers):
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
                        prev_layer_len = len(self.layers[layer_idx-1])
                        for prev_gate_idx in range(prev_layer_len):
                            if prev_gate_idx == input_idx:
                                G.add_edge(f"layer_{layer_idx-1}_gate_{prev_gate_idx}", node_id)
                                break

        # Add output node
        output_node = "output"
        G.add_node(output_node)
        pos[output_node] = (len(self.layers), 0)
        labels[output_node] = str(self.output_gate.operation)
        
        # Connect output gate to output node
        # last_layer_idx = len(self.layers) - 1
        # for gate_idx, gate in enumerate(self.layers[-1].gates):
        #     G.add_edge(f"layer_{last_layer_idx}_gate_{gate_idx}", output_node)
        for gate_idx in self.output_gate.input_idxs:
            G.add_edge(f"layer_{len(self.layers)-1}_gate_{gate_idx}", output_node)

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

