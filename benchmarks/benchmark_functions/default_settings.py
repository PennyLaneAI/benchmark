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
Benchmarks for a machine learning application.
"""
import networkx as nx

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

from numpy.random import random
from functools import partial
from pennylane.templates import BasicEntanglerLayers
from pennylane.templates.decorator import template as template_decorator
from pennylane.templates.subroutines import UCCSD
from pennylane import Identity, PauliX, PauliY, PauliZ

from .hamiltonians import ham_h2



def _core_defaults(hyperparams):
	"""Uses hyperparameters or defaults to construct the components of the circuit.

	Args:
		hyperparams (dict): hyperparameters provided by user
	"""
	# get hyperparameters or set default values
	n_wires = hyperparams.pop('n_wires', 4)
	n_layers = hyperparams.pop('n_layers', 6)
	interface = hyperparams.pop('interface', 'autograd')
	params = hyperparams.pop('params', random(size=(n_layers, n_wires)))
	measurement = hyperparams.pop('measurement', qml.expval(qml.PauliZ(0)))
	diff_method = hyperparams.pop('diff_method', 'best')
	device = hyperparams.pop('device', 'default.qubit')
	template = hyperparams.pop('template', None)

	# if device name is given, create device
	if isinstance(device, str):
		device = qml.device(device, wires=n_wires)

	# wrap default template so it only takes the parameters as argument
	if template is None:
		@template_decorator
		def Template(params_):
			BasicEntanglerLayers(params_, wires=range(n_wires))

		template = Template

	return device, diff_method, interface, params, template, measurement


def _vqe_defaults(hyperparams):
	"""Uses hyperparameters or defaults to construct the components of the VQE circuit for the
	hydrogen molecule with the sto-3g basis set.

	Args:
		hyperparams (dict): hyperparameters provided by user
	"""

	electrons = 2
	qubits = 4

	singles, doubles = qchem.excitations(electrons, qubits)
	s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
	hf_state = qchem.hf_state(electrons, qubits)
	ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)
	params = np.array([3.14545258, 3.13766988, -0.21446816])

	ham = hyperparams.pop('Hamiltonian', ham_h2)
	ansatz = hyperparams.pop('ansatz', ansatz)
	params = hyperparams.pop('params', params)
	n_steps = hyperparams.pop('n_steps', 1)
	device = hyperparams.pop('device', 'default.qubit')
	interface = hyperparams.pop('interface', 'autograd')
	diff_method = hyperparams.pop('diff_method', 'best')
	optimize = hyperparams.pop('optimize', True)

	options_dict = {'interface': interface, 'diff_method': diff_method, 'optimize': optimize}

	if isinstance(device, str):
		device = qml.device(device, wires=qubits)

	return ham, ansatz, params, n_steps, device, options_dict


def _qaoa_defaults(hyperparams):
	"""Uses hyperparameters or defaults to construct the components of the QAOA circuit for finding
	the minimum vertex cover of a graph.

	Args:
		hyperparams (dict): hyperparameters provided by user
	"""
	graph = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])

	graph = hyperparams.pop('graph', graph)
	n_layers = hyperparams.pop('n_layers', 2)
	params = hyperparams.pop('params', [[0.5] * n_layers, [0.5] * n_layers])
	device = hyperparams.pop('device', 'default.qubit')
	interface = hyperparams.pop('interface', 'autograd')
	diff_method = hyperparams.pop('diff_method', 'best')

	# if device name is given, create device
	if isinstance(device, str):
		device = qml.device(device, wires=len(graph.nodes), analytic=False)

	options_dict = {'interface': interface, 'diff_method': diff_method}

	return graph, n_layers, params, device, options_dict


def _ml_defaults(hyperparams):
	"""Uses hyperparameters or defaults to construct the components of the machine learning benchmark.

	Args:
		hyperparams (dict): hyperparameters provided by user
	"""
	# get hyperparameters or set default values
	n_features = hyperparams.pop('n_features', 4)
	n_samples = hyperparams.pop('n_samples', 20)
	interface = hyperparams.pop('interface', 'autograd')
	diff_method = hyperparams.pop('diff_method', 'best')
	device = hyperparams.pop('device', 'default.qubit')

	# if device name is given, create device
	if isinstance(device, str):
		device = qml.device(device, wires=n_features)

	# data
	x0 = np.random.normal(loc=-1, scale=1, size=(n_samples // 2, n_features))
	x1 = np.random.normal(loc=1, scale=1, size=(n_samples // 2, n_features))
	x = np.concatenate([x0, x1], axis=0)
	y = np.concatenate([-np.ones(50), np.ones(50)], axis=0)
	data = list(zip(x, y))

	return data, device, diff_method, interface
