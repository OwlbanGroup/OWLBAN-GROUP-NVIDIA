#!/usr/bin/env python
"""Test QSVM and VQC from qiskit-machine-learning"""
import os
os.environ['API_PASSWORD'] = 'testpassword123'

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import QSVC, VQC

# Create a simple quantum circuit
n_qubits = 4
feature_map = QuantumCircuit(n_qubits)
params = ParameterVector('x', n_qubits)
feature_map.append(params, range(n_qubits))

# Add some entanglement
for i in range(n_qubits - 1):
    feature_map.cx(i, i + 1)

# Create simulator
simulator = AerSimulator()

# Generate synthetic data
X = np.random.randn(20, 4)
y = np.random.randint(0, 2, 20)

# Test QSVC (Quantum Support Vector Classifier)
print("Testing QSVC...")
try:
    qsvc = QSVC(
        feature_map=feature_map,
        operator=feature_map,
        interpreter='all',
        solver='lbfgs',
        tol=1e-3,
        max_iter=100,
        quantum_kernel_algo='statevector',
        backend=simulator
    )
    print("QSVC created successfully")
    
    # Train
    print("Training QSVC...")
    qsvc.fit(X, y)
    print(f"QSVC trained successfully, training_time={qsvc.training_time}")
    
    # Predict
    X_test = np.random.randn(5, 4)
    predictions = qsvc.predict(X_test)
    print(f"QSVC predictions: {predictions}")
except Exception as e:
    print(f"QSVC failed: {type(e).__name__}: {e}")

print()

# Test VQC (Variational Quantum Classifier)
print("Testing VQC...")
try:
    vqc = VQC(
        feature_map=feature_map,
        optimizer='COBYLA',
        max_trials=5,
        n_qubits=n_qubits,
        device=simulator
    )
    print("VQC created successfully")
    
    # Train
    print("Training VQC...")
    vqc.fit(X, y)
    print(f"VQC trained successfully, training_time={vqc.training_time}")
    
    # Predict
    X_test = np.random.randn(5, 4)
    predictions = vqc.predict(X_test)
    print(f"VQC predictions: {predictions}")
except Exception as e:
    print(f"VQC failed: {type(e).__name__}: {e}")