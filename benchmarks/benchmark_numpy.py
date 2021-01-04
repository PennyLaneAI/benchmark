# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmark for evaluating a circuit consisting only of single-qubit
gates and nearest-neighbour two-qubit gates.
"""
from math import pi, ceil
import random

import numpy as np
import pennylane as qml


class NumpyCircuit:

	params = ([2, 5, 10], [1, 4, 8])
	param_names = ['n_qubits', 'n_layers']

	def time_nearest_neighbour(self, n_qubits, n_layers):
		"""Nearest-neighbour circuit benchmark.

		Creates a parametrized quantum circuit with n layers.
		Each layer consists of single-qubit parametrized rotations,
		and two-qubit gates between nearest-neighbour qubits.
		"""
		params1 = np.random.uniform(high=2 * pi, size=(n_layers, n_qubits))
		params2 = np.random.uniform(high=2 * pi, size=(n_layers, n_qubits - 1))
		all_wires = range(n_qubits)
		dev = qml.device('default.qubit', wires=n_qubits)

		@qml.qnode(dev)
		def circuit():
			for layer in range(n_layers):
				qml.broadcast(
					qml.RX, pattern="single", wires=all_wires, parameters=params1[layer],
				)
				qml.broadcast(
					qml.CRY, pattern="chain", wires=all_wires, parameters=params2[layer],
				)
				return qml.expval(qml.PauliZ(0))

		circuit()

	def time_iqp(self, n_qubits, n_layers):
		"""IQP circuit benchmark.
		"""
		random_iqp_wires = random.sample(range(n_qubits), ceil(min(2, n_qubits) * random.random()))
		dev = qml.device('default.qubit', wires=n_qubits)

		@qml.qnode(dev)
		def circuit():
			"""Mutable IQP quantum circuit."""
			for i in range(n_qubits):
				qml.Hadamard(i)

			for i in range(n_layers * n_qubits):
				wires = random_iqp_wires
				if len(wires) == 1:
					qml.PauliZ(wires=wires)
				elif len(wires) == 2:
					qml.CZ(wires=wires)
				elif len(wires) == 3:
					qml.CCZ(wires)

			for i in range(n_qubits):
				qml.Hadamard(i)

			return qml.expval(qml.PauliZ(0))

		circuit()


class NumpyGradient:

	params = ([2, 5, 10], [5, 20, 40])
	param_names = ['n_qubits', 'n_params']

	def time_rotations(self, n_qubits, n_params):
		"""Rotations gradient benchmark.

		The circuit consists of rotations applied to consecutive wires,
		starting from wire 0 after the last wire is reached.
		"""
		params = np.random.uniform(high=2 * pi, size=n_params)
		dev = qml.device('default.qubit', wires=n_qubits)

		@qml.qnode(dev)
		def circuit(params):
			for i, w in enumerate(params):
				qml.RX(w, wires=i % n_qubits)
			return qml.expval(qml.PauliZ(wires=n_qubits-1))

		circuit.jacobian([params])
