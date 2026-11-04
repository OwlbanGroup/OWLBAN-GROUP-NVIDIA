#!/usr/bin/env python
"""Test Qiskit AerSimulator setup"""
import os
os.environ['API_PASSWORD'] = 'testpassword123'

from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler

# Test creating and using a simulator
simulator = AerSimulator()
print(f'AerSimulator created: {simulator}')

# Test simple circuit
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sampler = StatevectorSampler()
result = sampler.run([qc], shots=1000)
print(f'Sampler result: {result}')