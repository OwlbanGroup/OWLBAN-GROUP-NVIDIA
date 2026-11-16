import numpy as np

from quantum_software import GroverSearch, QuantumCircuit, QuantumSoftware


def test_hadamard_creates_balanced_superposition():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    probabilities = circuit.probabilities()
    np.testing.assert_allclose(probabilities, np.array([0.5, 0.5]), atol=1e-8)


def test_bell_pair_creates_entangled_state():
    software = QuantumSoftware(qubit_count=2)
    circuit = software.bell_pair()
    probs = circuit.probabilities()
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)
    assert probs.shape == (4,)


def test_grover_search_identifies_best_index():
    search = GroverSearch(8, [3])
    result = search.run(iterations=2)
    assert result["winner"] == 3
    assert np.isclose(result["probabilities"].sum(), 1.0)


def test_quantum_software_runs_and_evaluates():
    software = QuantumSoftware(qubit_count=2)
    result = software.run_algorithm(shots=200)
    assert result.measured_bitstring
    assert len(result.state_vector) == 4
    assert result.shot_count == 200

    evaluation = software.evaluate([0.1, 0.8, 0.9, 0.2])
    assert evaluation["index"] == 2
    assert evaluation["score"] == 0.9

    grover = software.grover_search([0.1, 0.8, 0.9, 0.2], iterations=2)
    assert grover["best_index"] == 2
