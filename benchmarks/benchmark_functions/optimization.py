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
Benchmarks for a circuit training application.
"""
from numpy.random import random

import pennylane as qml
import tensorflow as tf
import torch
from pennylane import numpy as pnp
from .default_settings import _core_defaults


def benchmark_optimization(hyperparams={}, n_steps=20, num_repeats=1):
	"""Trains a quantum circuit for n_steps steps with a gradient descent optimizer.

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

	n_steps (int): number of optimization steps
	num_repeats (int): How often the same circuit is evaluated in a for loop. Default is 1.
	"""

	device, diff_method, interface, params, template, measurement = _core_defaults(hyperparams)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params_):
		template(params_)
		measurement.queue()
		return measurement

	for _ in range(num_repeats):

		if interface == 'autograd':
			params = pnp.array(params, requires_grad=True)
			opt = qml.GradientDescentOptimizer(stepsize=0.1)

			for i in range(n_steps):
				params = opt.step(circuit, params)

		elif interface == 'tf':
			params = tf.Variable(params)
			opt = tf.keras.optimizers.SGD(learning_rate=0.1)

			for i in range(n_steps):
				with tf.GradientTape() as tape:
					loss = circuit(params)
				gradients = tape.gradient(loss, [params])
				opt.apply_gradients(zip(gradients, [params]))

		elif interface == 'torch':
			params = torch.tensor(params, requires_grad=True)
			opt = torch.optim.SGD([params], lr=0.1)

			def closure():
				opt.zero_grad()
				loss = circuit(params)
				loss.backward()
				return loss

			for i in range(n_steps):
				opt.step(closure)

	# TODO: jax