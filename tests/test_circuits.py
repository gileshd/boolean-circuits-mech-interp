import numpy as np
import pytest

from boolean_circuits.circuits import AND, OR, NOT, NAND, NOR, XOR, Gate, Layer, Circuit


def assert_all_equal(x, y):
    assert np.all(x == y)


def test_AND():
    and_op = AND()
    assert and_op(np.array([True, True])) == True
    assert and_op(np.array([True, False])) == False
    assert and_op(np.array([False, True])) == False
    assert and_op(np.array([False, False])) == False
    assert and_op(np.array([True, True, True])) == True
    assert and_op(np.array([True, False, True])) == False
    assert and_op(np.array([False, True, True])) == False
    assert and_op(np.array([False, False, True])) == False
    with pytest.raises(ValueError):
        # Raise error if less than two inputs
        and_op(np.array([True]))


def test_OR():
    or_op = OR()
    assert or_op(np.array([True, True])) == True
    assert or_op(np.array([True, False])) == True
    assert or_op(np.array([False, True])) == True
    assert or_op(np.array([False, False])) == False
    assert or_op(np.array([True, True, True])) == True
    assert or_op(np.array([True, False, True])) == True
    assert or_op(np.array([False, False, False])) == False
    with pytest.raises(ValueError):
        # Raise error if less than two inputs
        or_op(np.array([True]))


def test_NOT():
    not_op = NOT()
    assert not_op(np.array([True])) == False
    assert not_op(np.array([False])) == True
    with pytest.raises(ValueError):
        # Raise error if more than one input
        not_op(np.array([True, False]))


def test_NAND():
    nand_op = NAND()
    assert nand_op(np.array([True, True])) == False
    assert nand_op(np.array([True, False])) == True
    assert nand_op(np.array([False, True])) == True
    assert nand_op(np.array([False, False])) == True
    assert nand_op(np.array([True, True, True])) == False
    assert nand_op(np.array([True, False, True])) == True
    with pytest.raises(ValueError):
        # Raise error if less than two inputs
        nand_op(np.array([True]))


# this is probably just overcomplicating things...
@pytest.mark.parametrize(
    "input_values",
    [
        np.array([True, True]),
        np.array([True, False]),
        np.array([False, True]),
        np.array([False, False]),
        np.array([True, True, True]),
        np.array([True, False, True]),
    ],
)
def test_NOR(input_values):
    nor_op = NOR()
    not_op = NOT()
    or_op = OR()
    assert nor_op(input_values) == not_op(np.array([or_op(input_values)]))
    with pytest.raises(ValueError):
        # Raise error if less than two inputs
        nor_op(np.array([True]))


def test_XOR():
    xor_op = XOR()
    assert xor_op(np.array([True, True])) == False
    assert xor_op(np.array([True, False])) == True
    assert xor_op(np.array([False, True])) == True
    assert xor_op(np.array([False, False])) == False
    assert xor_op(np.array([True, True, True])) == True
    assert xor_op(np.array([True, False, True])) == False
    assert xor_op(np.array([False, False, False])) == False
    assert xor_op(np.array([True, True, False])) == False
    assert xor_op(np.array([True, False, False])) == True
    assert xor_op(np.array([False, True, False])) == True
    assert xor_op(np.array([False, False, True])) == True
    with pytest.raises(ValueError):
        # Raise error if less than two inputs
        xor_op(np.array([True]))


def test_Gate():
    gate = Gate(AND(), np.array([0, 1]))
    assert gate(np.array([True, True])) == True
    assert gate(np.array([True, True, False])) == True
    assert gate(np.array([True, False, True])) == False


class TestLayer:
    layer = Layer(
        [
            Gate(AND(), np.array([0, 1])),
            Gate(OR(), np.array([2, 3])),
            Gate(NOT(), np.array([4])),
        ]
    )

    def test_call(self):
        assert_all_equal(
            self.layer(np.array([True, True, False, True, False])),
            np.array([True, True, True]),
        )

    def test_len(self):
        assert len(self.layer) == 3

    def test_max_input_idx(self):
        assert self.layer._max_input_idx == 4

    def test_check_idxs_present(self):
        prev_layer = Layer([Gate(AND(), np.array([0, 1]))])
        with pytest.raises(ValueError):
            self.layer._check_idxs_present(prev_layer)


class TestCircuit:
    layer1 = Layer(
        [
            Gate(AND(), np.array([0, 1])),
            Gate(OR(), np.array([2, 3])),
            Gate(NOT(), np.array([4])),
        ]
    )
    layer2 = Layer(
        [
            Gate(AND(), np.array([0, 1])),
            Gate(XOR(), np.array([1, 2])),
        ]
    )
    output_gate = Gate(XOR(), np.array([0, 1]))
    circuit = Circuit([layer1, layer2], output_gate)

    @pytest.mark.parametrize(
        "input_values,expected_output,expected_intermediate",
        [
            (
                np.array([True, True, False, True, False]),
                True,
                [np.array([True, True, True]), np.array([True, False])],
            ),
            (
                np.array([True, True, False, True, True]),
                False,
                [np.array([True, True, False]), np.array([True, True])],
            ),
            (
                np.array([True, False, False, False, True]),
                False,
                [np.array([False, False, False]), np.array([False, False])],
            ),
        ],
    )
    def test_call(self, input_values, expected_output, expected_intermediate):
        output, intermediate = self.circuit(input_values)
        assert output == expected_output
        for i, e in zip(intermediate, expected_intermediate):
            assert_all_equal(i, e)

    def test_size(self):
        assert self.circuit.size == 6

    def test_check_wiring(self):
        self.circuit._check_wiring()
        layer1_too_small = Layer([Gate(AND(), np.array([0, 1]))])
        layer2_expects_bigger = Layer([Gate(AND(), np.array([0, 1]))])
        output_gate = Gate(NOT(), np.array([0]))
        with pytest.raises(ValueError):
            circuit_bad_wiring = Circuit(
                [layer1_too_small, layer2_expects_bigger], output_gate
            )
