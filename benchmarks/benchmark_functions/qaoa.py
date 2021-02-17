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
from pennylane import qaoa
from .default_settings import _qaoa_defaults
from pennylane import numpy as np


def benchmark_qaoa(hyperparams={}):
	"""
	Performs QAOA optimizations.

	Args:
		hyperparams (dict): hyperparameters to configure this benchmark

			* 'graph': Graph represented as a NetworkX Graph class

			* 'n_layers': Number of layers in the QAOA circuit

			* 'params': Numpy array of trainable parameters that is fed into the circuit

			* 'n_steps': Number of QAOA steps

			* 'device': Device on which the circuit is run

			* 'interface': Name of the interface to use

			* 'diff_method': Name of differentiation method
	"""

	graph, n_layers, params, n_steps, device, options_dict = _qaoa_defaults(hyperparams)

	H_cost, H_mixer = qaoa.min_vertex_cover(graph, constrained=False)

	n_wires = 4

	def qaoa_layer(gamma, alpha):
		qaoa.cost_layer(gamma, H_cost)
		qaoa.mixer_layer(alpha, H_mixer)

	def comp_basis_measurement(wires):
		n_wires = len(wires)
		return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)

	@qml.qnode(device)
	def circuit(params):
		for w in range(n_wires):
			qml.Hadamard(wires=w)
		qml.layer(qaoa_layer, n_layers, params[0], params[1])
		return qml.sample(comp_basis_measurement(range(n_wires)))

	circuit(params)
