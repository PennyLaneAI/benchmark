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
Benchmarks for simple circuit evaluations.
"""
import pennylane as qml
import tensorflow as tf
import torch
from pennylane import numpy as pnp
from .default_settings import _set_defaults

qml.enable_tape()


def benchmark_gradient(hyperparams={}, num_repeats=1):
	"""Computes the gradient of a quantum circuit.

	Unless otherwise specified by the hyperparameters, the circuit consists of 6 layers of `BasicEntanglerLayers`
	run on 4 qubits, followed by measuring the Pauli-Z observable of the first wire, and is run using
	parameter shift differentiation, a 'default.qubit' device and the autograd interface.

		Args:
		hyperparams (dict): hyperparameters to configure this benchmark

			* 'n_wires': Number of wires to use. Will be ignored if custom device and template are provided.

			* 'n_layers': Number of layers in the default template. Will be ignored if custom params are provided.

			* 'diff_method': name of differentiation method

			* 'device': device on which the circuit is run, or valid device name

			* 'interface': name of the interface to use

			* 'template': Template to use. The template must take the trainable parameters as its only argument.

			* 'params': Numpy array of trainable parameters that is fed into the template.

			* 'measurement': measurement function like `qml.expval(qml.PauliZ(0)))`

		num_repeats (int): How often the same circuit is evaluated in a for loop. Default is 1.
	"""

	device, diff_method, interface, params, template, measurement = _set_defaults(hyperparams)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params_):
		template(params_)
		measurement.queue()
		return measurement

	for _ in range(num_repeats):

		if interface == 'autograd':
			params = pnp.array(params, requires_grad=True)
			jac = qml.jacobian(circuit)
			jac(params)

		elif interface == 'tf':
			params = tf.Variable(params)
			with tf.GradientTape() as tape:
				result = circuit(params)
			tape.gradient(result, [params])

		elif interface == 'torch':
			params = torch.tensor(params, requires_grad=True)
			result = circuit(params)
			result.backward()

		# TODO: jax
