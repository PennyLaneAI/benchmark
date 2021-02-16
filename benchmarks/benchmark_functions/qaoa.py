# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Benchmarks for QAOA optimizations.
"""
import pennylane as qml
from pennylane import numpy as np
from .default_settings import _qaoa_defaults


def benchmark_qaoa(hyperparams={}):
	"""
	Performs QAOA optimizations.

	Args:
		hyperparams (dict): hyperparameters to configure this benchmark

			* 'graph': Graph represented as a NetworkX Graph class

			* 'n_layers': Number of layers in the QAOA circuit

			* 'params': Numpy array of trainable parameters that is fed into the circuit

			* 'n_steps': Number of QAOA steps

			* 'device': Device on which the circuit is run, or valid device name

			* 'interface': Name of the interface to use

			* 'diff_method': Name of differentiation method
	"""

	graph, n_layers, params, n_steps, device, options_dict = _qaoa_defaults(hyperparams)

	n_wires = len(graph)

	def U_B(beta):
		for wire in range(n_wires):
			qml.RX(2 * beta, wires=wire)

	def U_C(gamma):
		for edge in graph:
			wire1 = edge[0]
			wire2 = edge[1]
			qml.CNOT(wires=[wire1, wire2])
			qml.RZ(gamma, wires=wire2)
			qml.CNOT(wires=[wire1, wire2])

	def comp_basis_measurement(wires):
		return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)

	@qml.qnode(device)
	def circuit(gammas, betas, n_layers):
		for wire in range(n_wires):
			qml.Hadamard(wires=wire)
		for i in range(n_layers):
			U_C(gammas[i])
			U_B(betas[i])
		return qml.sample(comp_basis_measurement(range(n_wires)))

	for _ in range(n_steps):
		circuit(params[0], params[1], n_layers=n_layers)
