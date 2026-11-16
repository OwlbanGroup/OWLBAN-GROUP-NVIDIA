"""Quantum algorithm helpers for the OWLBAN quantum software package."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


class GroverSearch:
    """A compact Grover-style amplitude amplification implementation.

    This is intentionally lightweight and deterministic so it works in a CPU-only
    environment while teaching the same core principles as a full quantum search
    algorithm.
    """

    def __init__(self, search_space_size: int, marked_indices: Sequence[int] | None = None):
        if search_space_size <= 0:
            raise ValueError("search_space_size must be positive")
        self.search_space_size = int(search_space_size)
        self.marked_indices = list(marked_indices or [0])

    def _oracle(self, state: np.ndarray) -> np.ndarray:
        result = state.copy()
        for idx in self.marked_indices:
            if 0 <= idx < len(state):
                result[idx] = -result[idx]
        return result

    def _diffusion(self, state: np.ndarray) -> np.ndarray:
        mean = np.mean(state)
        return 2 * mean - state

    def run(self, iterations: int = 1) -> dict:
        state = np.full(self.search_space_size, 1.0 / np.sqrt(self.search_space_size), dtype=np.complex128)
        for _ in range(max(0, iterations)):
            state = self._oracle(state)
            state = self._diffusion(state)
        probabilities = np.abs(state) ** 2
        winner = int(np.argmax(probabilities))
        return {
            "winner": winner,
            "state": state,
            "probabilities": probabilities,
            "marked_indices": list(self.marked_indices),
        }


class QuantumOptimizer:
    """Simple optimizer that uses Grover-style amplification for a discrete search."""

    def __init__(self, values: Sequence[float]):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            raise ValueError("values must not be empty")
        self.values = values

    def optimize(self, iterations: int = 1) -> dict:
        search_space_size = self.values.size
        best_index = int(np.argmax(self.values))
        marked = [best_index]
        grover = GroverSearch(search_space_size, marked)
        result = grover.run(iterations=iterations)
        return {
            "best_index": int(np.argmax(self.values)),
            "best_value": float(self.values[best_index]),
            "winner": int(result["winner"]),
            "probabilities": result["probabilities"],
        }
