"""Lightweight quantum software simulation engine.

This module provides a small but useful quantum programming toolkit that does not
require external quantum frameworks such as Qiskit. It is designed to run in a
plain CPU environment while still exposing the core concepts that a quantum
application needs: circuits, state evolution, measurement, and optimization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class QuantumResult:
    """Result of a quantum circuit execution."""

    state_vector: np.ndarray
    probabilities: np.ndarray
    measured_bitstring: str = ""
    shot_count: int = 1
    expectation: float = 0.0


class QuantumCircuit:
    """A simple gate-based quantum circuit simulator."""

    def __init__(self, qubit_count: int = 1):
        if qubit_count <= 0:
            raise ValueError("qubit_count must be positive")
        self.qubit_count = qubit_count
        self.state = np.zeros(2 ** qubit_count, dtype=np.complex128)
        self.state[0] = 1.0
        self.gates: List[Tuple[str, Tuple[int, ...], float]] = []

    def _basis_index(self, indices: Sequence[int]) -> int:
        if not indices:
            raise ValueError("At least one qubit index is required")
        for idx in indices:
            if idx < 0 or idx >= self.qubit_count:
                raise IndexError(f"Qubit index {idx} is out of range for {self.qubit_count} qubits")
        return 0

    def apply_gate(self, gate: str, *targets: int, phase: float = 0.0) -> "QuantumCircuit":
        """Apply a gate to one or more qubits."""
        normalized = gate.upper()
        if normalized in {"H", "X", "Y", "Z", "RX", "RY", "RZ", "CX", "CZ", "I"}:
            self.gates.append((normalized, tuple(targets), phase))
            self._apply_registered_gate(normalized, tuple(targets), phase)
            return self
        raise ValueError(f"Unsupported gate: {gate}")

    def _apply_registered_gate(self, gate: str, targets: Tuple[int, ...], phase: float) -> None:
        if gate == "I":
            return
        if gate in {"H", "X", "Y", "Z"}:
            self._apply_single_qubit_gate(gate, targets[0])
            return
        if gate in {"RX", "RY", "RZ"}:
            self._apply_rotation_gate(gate, targets[0], phase)
            return
        if gate in {"CX", "CZ"}:
            self._apply_two_qubit_gate(gate, targets[0], targets[1])
            return
        raise ValueError(f"Gate not implemented: {gate}")

    def _apply_single_qubit_gate(self, gate: str, target: int) -> None:
        bit_mask = 1 << target
        new_state = np.zeros_like(self.state)
        for idx, amplitude in enumerate(self.state):
            if idx & bit_mask:
                continue
            paired = idx | bit_mask
            a = self.state[idx]
            b = self.state[paired]
            if gate == "H":
                new_state[idx] = (a + b) / np.sqrt(2)
                new_state[paired] = (a - b) / np.sqrt(2)
            elif gate == "X":
                new_state[idx] = b
                new_state[paired] = a
            elif gate == "Y":
                new_state[idx] = -1j * b
                new_state[paired] = 1j * a
            elif gate == "Z":
                new_state[idx] = a
                new_state[paired] = -b
        self.state = new_state

    def _apply_rotation_gate(self, gate: str, target: int, angle: float) -> None:
        bit_mask = 1 << target
        new_state = self.state.copy()
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        for idx in range(len(self.state)):
            if idx & bit_mask:
                continue
            paired = idx | bit_mask
            if gate == "RX":
                new_state[idx] = c * self.state[idx] - 1j * s * self.state[paired]
                new_state[paired] = -1j * s * self.state[idx] + c * self.state[paired]
            elif gate == "RY":
                new_state[idx] = c * self.state[idx] - s * self.state[paired]
                new_state[paired] = s * self.state[idx] + c * self.state[paired]
            elif gate == "RZ":
                phase = np.exp(1j * angle / 2)
                new_state[idx] = phase * self.state[idx]
                new_state[paired] = np.exp(-1j * angle / 2) * self.state[paired]
        self.state = new_state

    def _apply_two_qubit_gate(self, gate: str, control: int, target: int) -> None:
        if control == target:
            raise ValueError("Control and target qubits must be different for two-qubit gates")
        new_state = np.zeros_like(self.state)
        for idx, amplitude in enumerate(self.state):
            if gate == "CX":
                if ((idx >> control) & 1) == 1:
                    flipped_idx = idx ^ (1 << target)
                    new_state[flipped_idx] += amplitude
                else:
                    new_state[idx] += amplitude
            elif gate == "CZ":
                if ((idx >> control) & 1) == 1 and ((idx >> target) & 1) == 1:
                    new_state[idx] += -amplitude
                else:
                    new_state[idx] += amplitude
        self.state = new_state

    def h(self, target: int) -> "QuantumCircuit":
        return self.apply_gate("H", target)

    def x(self, target: int) -> "QuantumCircuit":
        return self.apply_gate("X", target)

    def z(self, target: int) -> "QuantumCircuit":
        return self.apply_gate("Z", target)

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        return self.apply_gate("CX", control, target)

    def measure(self, shots: int = 1, rng: Optional[random.Random] = None) -> QuantumResult:
        """Measure all qubits in the computational basis."""
        probabilities = np.abs(self.state) ** 2
        total = probabilities.sum()
        if total == 0:
            probabilities = np.full_like(probabilities, 1.0 / len(probabilities))
        else:
            probabilities = probabilities / total
        if shots <= 0:
            raise ValueError("shots must be positive")

        random_state = rng or random.Random()
        outcomes = []
        for _ in range(shots):
            outcomes.append(self._sample_once(probabilities, random_state))

        sampled_counts = {i: outcomes.count(i) for i in range(len(probabilities))}
        most_likely_index = max(sampled_counts, key=sampled_counts.get)
        measured = format(most_likely_index, f"0{self.qubit_count}b")

        if shots == 1:
            expectation = 0.0
        else:
            bit_matrix = np.array(
                [[int(bit) for bit in format(sample, f"0{self.qubit_count}b")] for sample in outcomes],
                dtype=float,
            )
            expectation = float(bit_matrix.mean())

        return QuantumResult(
            state_vector=self.state.copy(),
            probabilities=probabilities,
            measured_bitstring=measured,
            shot_count=shots,
            expectation=expectation,
        )

    def _sample_once(self, probabilities: np.ndarray, rng: random.Random) -> int:
        threshold = rng.random() * probabilities.sum()
        cumulative = 0.0
        for idx, probability in enumerate(probabilities):
            cumulative += float(probability)
            if cumulative >= threshold:
                return idx
        return len(probabilities) - 1

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def create_bell_pair(self) -> "QuantumCircuit":
        self.h(0)
        self.cx(0, 1)
        return self

    def __repr__(self) -> str:
        return f"QuantumCircuit(qubits={self.qubit_count}, gates={len(self.gates)})"


class QuantumSimulator:
    """Thin wrapper around a circuit and its execution logic."""

    def __init__(self):
        self.last_result: Optional[QuantumResult] = None

    def run(self, circuit: QuantumCircuit, shots: int = 100) -> QuantumResult:
        self.last_result = circuit.measure(shots=shots)
        return self.last_result

    def probability_distribution(self, circuit: QuantumCircuit) -> np.ndarray:
        return circuit.probabilities()


class QuantumSoftware:
    """High-level application layer for quantum workflows."""

    def __init__(self, qubit_count: int = 2):
        self.qubit_count = qubit_count
        self.simulator = QuantumSimulator()

    def bell_pair(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qubit_count)
        circuit.create_bell_pair()
        return circuit

    def optimize_binary(self, values: Sequence[float], target: Optional[float] = None) -> Dict[str, float]:
        """Simple objective optimization over a binary bitstring search space."""
        if not values:
            return {"best_value": 0.0, "best_index": 0}
        arr = np.asarray(values, dtype=float)
        best_index = int(np.argmax(arr))
        best_value = float(arr[best_index])
        if target is not None:
            best_value = min(best_value, float(target))
        return {"best_value": best_value, "best_index": best_index}

    def grover_search(self, values: Sequence[float], iterations: int = 1) -> Dict[str, float]:
        """Perform a lightweight Grover-style search over a discrete objective list."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"best_value": 0.0, "best_index": 0, "winner": 0}
        best_index = int(np.argmax(arr))
        from .algorithms import GroverSearch
        search = GroverSearch(arr.size, [best_index])
        result = search.run(iterations=max(1, int(iterations)))
        winner = int(result["winner"])
        return {
            "best_value": float(arr[best_index]),
            "best_index": best_index,
            "winner": winner,
            "winner_value": float(arr[winner]),
        }

    def run_algorithm(self, circuit: Optional[QuantumCircuit] = None, shots: int = 100) -> QuantumResult:
        active_circuit = circuit or self.bell_pair()
        result = self.simulator.run(active_circuit, shots=shots)
        return result

    def evaluate(self, values: Sequence[float]) -> Dict[str, float]:
        optimized = self.optimize_binary(values)
        return {
            "score": float(optimized["best_value"]),
            "index": int(optimized["best_index"]),
            "quality": float(np.mean(np.asarray(values, dtype=float))) if values else 0.0,
        }
