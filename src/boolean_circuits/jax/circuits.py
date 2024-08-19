from abc import abstractmethod
import equinox as eqx
import itertools
import jax
from jax import numpy as jnp
from jax import jit, vmap

from jaxtyping import Array, Bool, Int

# TOOD: Add docstrings
# TODO: (Equinox?)


class Operation:
    @abstractmethod
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        """Perform the operation on the input values."""
        raise NotImplementedError("Subclasses should implement this method")

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()


class AND(Operation):
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return jnp.all(input_values)


class OR(Operation):
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return jnp.any(input_values)


class NOT(Operation):
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        if len(input_values) != 1:
            raise ValueError("NOT operation takes exactly one input")
        # return jnp.bitwise_not(input_values)
        return input_values[0] == False


class NAND(Operation):
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return jnp.bitwise_not(jnp.all(input_values))


class NOR(Operation):
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return jnp.bitwise_not(jnp.any(input_values))


class XOR(Operation):
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        if len(input_values) < 2:
            raise ValueError("Operation requires at least two inputs")
        return jnp.sum(input_values) % 2 == 1


class NOOP(Operation):
    def __call__(self, input_values: Bool[Array, "op_dim"]) -> Bool:
        # TODO: It would be nice to allow output dim > 1 but that complicates layer size calculattions...
        if len(input_values) != 1:
            raise ValueError("NOOP operation takes exactly one input")
        return input_values


class Gate(eqx.Module):
    operation: Operation
    input_idxs: Int[Array, "op_dim"]

    def __call__(self, input_values: Bool[Array, "data_dim"]) -> Bool:
        """Call the gate operation on `input_idxs` within `input values`."""
        return self.operation(input_values[self.input_idxs])

    def __str__(self) -> str:
        return f"{self.operation}({self.input_idxs})"

    def __repr__(self) -> str:
        return self.__str__()


class Layer(eqx.Module):
    gates: list[Gate]

    def __call__(
        self, input_values: Bool[Array, "data_dim"]
    ) -> Bool[Array, "{len(self.gates)}"]:
        """Call each gate in the layer and combine the outputs into an array."""
        return jnp.array([gate(input_values) for gate in self.gates], dtype=bool)

    def __str__(self) -> str:
        return "; ".join([str(gate) for gate in self.gates])

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.gates)

    @property
    def _max_input_idx(self) -> int:
        """The maximum input index found in a gate in the layer."""
        return max(max(gate.input_idxs) for gate in self.gates)

    def _check_idxs_present(self, prev_layer) -> None:
        """Check that all gate indices are present in the `prev_layer`."""
        max_idx = self._max_input_idx
        if max_idx > len(prev_layer.gates) - 1:
            raise ValueError("Not all gate indices not present in previous layer")


class Circuit(eqx.Module):
    layers: list[Layer]
    # TODO: Do I really need to limit this to a single output gate?
    #       This might be more flexible if this is just a final layer.
    output_gate: Gate
    input_size: int

    def __init__(self, layers: list[Layer], output_gate: Gate, input_size=None):
        self.layers = layers
        self.output_gate = output_gate
        self.input_size = (
            self.layers[0]._max_input_idx + 1 if input_size is None else input_size
        )
        self._check_wiring()

    def __call__(
        self, input_values: Bool[Array, "{self.input_size}"]
    ) -> tuple[Bool[Array, "output_dim"], Bool[Array, "{self.size-1}"]]:
        intermediate_values = []
        for layer in self.layers:
            input_values = layer(input_values)
            intermediate_values.append(input_values.astype(int))
        output = self.output_gate(input_values)
        return jnp.array(output.astype(int)), jnp.concatenate(intermediate_values, axis=-1)

    def __str__(self) -> str:
        output = "\n".join([str(layer) for layer in self.layers])
        return output + f"\n{self.output_gate}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def size(self) -> int:
        """Return the number of gates in the circuit."""
        return sum(len(layer) for layer in self.layers) + 1

    def unflatten_values(self, values):
        """Unflatten the values array into a dictionary of layer outputs."""
        d = {}
        i = 0
        for l, layer in enumerate(self.layers):
            d[f"layer_{l}"] = {}
            for n, _ in enumerate(layer.gates):
                d[f"layer_{l}"][f"gate_{n}"] = values[i]
                i += 1
        return d

    # TODO: Maybe I should be normalising the gate influence differenty?
    #       I think a more natural way to normalise the gate influence might be
    #       to only consider bit flips where the bit is "upstream" of the gate?
    def calculate_influences(self):
        """
        Calculate the influence of each input bit and each gate on the output.

        The influence of an input bit is defined as the probability that flipping that bit will 
        change the output of the circuit. This is a standard definition. 

        The influence of a gate is defined as the probability that flipping an input bit will
        change both the ouput of that gate and the output of the circuit. As far as I am aware this
        is not a standard definition. 
         An alternative definition would be the probability that flipping the gate output will 
        change the circuit output. However this does not reflect the sensitivity of the gate to 
        the input.

        Returns:
            tuple[Array(n_bits,), Array(n_gates,)] - A tuple containing the influence of each input 
            bit and each gate.
        """
        x = jnp.array(list(itertools.product([0, 1], repeat=self.input_size)))
        n_bits = x.shape[-1]

        _add_trees = lambda t1, t2: jax.tree.map(jnp.add, t1, t2)
        _flip_bit = lambda x, bit: x.at[bit].set(1 - x[bit])

        def _check_flip(x, bit):
            """Check the influence of flipping a single bit."""
            output, intermediates = self(x)

            flipped_output, flipped_intermediates = self(_flip_bit(x, bit))
            output_changed = (flipped_output != output).any()

            intermediates_changed = intermediates != flipped_intermediates
            int_and_out_changed = output_changed & intermediates_changed

            return output_changed, int_and_out_changed

        @jit
        def body_func(carry, x):
            """Update carry with the influence of flipping each bit in x."""
            # Calculate the influence of flipping each bit in the input
            outputs_changed, intermediates_changed = vmap(_check_flip, in_axes=(None, 0))(x, jnp.arange(n_bits))
            # Average gate influence over bits
            intermediates_changed = intermediates_changed.mean(0) 
            # Repack the results
            updates = (outputs_changed, intermediates_changed)
            return _add_trees(carry, updates), None

        init_carry = (jnp.zeros(x.shape[-1]), jnp.zeros(self.size - 1))
        influences, _ = jax.lax.scan(body_func, init_carry, x) # (bit_influences, gate_influences)

        N_samples = x.shape[0]
        influences = jax.tree.map(lambda a: a / N_samples, influences) # Average over samples
        return influences


    def _check_wiring(self):
        """Ensure that no gate is referring to an input that doesn't exist."""
        if max(self.output_gate.input_idxs) > len(self.layers[-1]):
            raise ValueError
        # TODO: Make this not awful - add reference to layer number in error message.
        for layer, prev_layer in zip(self.layers[-1:0:-1], self.layers[-2::-1]):
            layer._check_idxs_present(prev_layer)
