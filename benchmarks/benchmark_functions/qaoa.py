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

	H_cost, H_mixer = qaoa.min_vertex_cover(graph, constrained=False)

	wires = range(len(params[0]))

	def qaoa_layer(gamma, alpha):
		qaoa.cost_layer(gamma, H_cost)
		qaoa.mixer_layer(alpha, H_mixer)

	def circuit(params, **kwargs):
		for w in wires:
			qml.Hadamard(wires=w)
		qml.layer(qaoa_layer, n_layers, params[0], params[1])

	cost_fn = qml.ExpvalCost(circuit, H_cost, device, **options_dict)

	opt = qml.GradientDescentOptimizer(stepsize=0.4)

	for n in range(n_steps):
		params = opt.step(cost_fn, params)

	@qml.qnode(device)
	def probability_circuit(gamma, alpha):
		circuit([gamma, alpha])
		return qml.probs(wires=wires)

	probs = probability_circuit(params[0], params[1])
