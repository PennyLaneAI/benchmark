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
from numpy.random import random

import pennylane as qml
from pennylane import numpy as pnp
from .default_settings import _ml_defaults


def _machine_learning_autograd(quantum_model, data):
	"""ML example with autograd interface."""

	def hybrid_model(x, w_quantum, w_classical):
		transformed_x = pnp.dot(w_classical, x)
		return quantum_model(transformed_x, w_quantum)

	def average_loss(w_quantum, w_classical):
		c = 0
		for x, y in data:
			prediction = hybrid_model(x, w_quantum, w_classical)
			c += (prediction - y) ** 2
		return c / len(data)

	n_features = len(data[0][0])

	w_quantum = pnp.array(random(size=(n_features, n_features)), requires_grad=True)
	w_classical = pnp.array(random(size=(n_features, n_features)), requires_grad=True)

	gradient_fn_wq = qml.grad(average_loss, argnum=0)
	gradient_fn_wc = qml.grad(average_loss, argnum=1)

	for i in range(50):
		w_quantum = w_quantum - 0.05 * gradient_fn_wq(w_quantum, w_classical)
		w_classical = w_classical - 0.05 * gradient_fn_wc(w_quantum, w_classical)


def _machine_learning_tf(quantum_model, data):
	"""ML example with tensorflow interface."""

	import tensorflow as tf
	data = [[tf.constant(x, dtype=tf.double), tf.constant(y, dtype=tf.double)] for x, y in data]

	def hybrid_model(x, w_quantum, w_classical):
		transformed_x = tf.linalg.matvec(w_classical, x)
		return quantum_model(transformed_x, w_quantum)

	def average_loss(w_quantum, w_classical):
		c = tf.constant(0, dtype=tf.double)
		for x, y in data:
			prediction = hybrid_model(x, w_quantum, w_classical)
			c = c + (prediction - y) ** 2
		return c / len(data)

	n_features = len(data[0][0])

	w_quantum = tf.Variable(random(size=(n_features, n_features)), dtype=tf.double)
	w_classical = tf.Variable(random(size=(n_features, n_features)), dtype=tf.double)

	for i in range(50):

		with tf.GradientTape() as tape:
			loss = average_loss(w_quantum, w_classical)

		grad_qu, grad_class = tape.gradient(loss, [w_quantum, w_classical])

		w_quantum = w_quantum - 0.05 * grad_qu
		w_classical = w_classical - 0.05 * grad_class
		print(w_quantum)
		print(w_classical)
		print(average_loss(w_quantum, w_classical))


def _machine_learning_torch(quantum_model, data):
	"""ML example with torch interface."""

	import torch
	data = [[torch.tensor(x), torch.tensor(y)] for x, y in data]

	def hybrid_model(x, w_quantum, w_classical):
		transformed_x = torch.matmul(w_classical, x)
		return quantum_model(transformed_x, w_quantum)

	def average_loss(w_quantum, w_classical):
		c = torch.tensor(0)
		for x, y in data:
			prediction = hybrid_model(x, w_quantum, w_classical)
			c += (prediction - y) ** 2
		return c / len(data)

	n_features = len(data[0][0])

	w_quantum = torch.tensor(random(size=(n_features, n_features)), requires_grad=True)
	w_classical = torch.tensor(random(size=(n_features, n_features)), requires_grad=True)

	for i in range(50):
		average_loss.backward()

		w_quantum = w_quantum - 0.05 * w_quantum.grad
		w_classical = w_classical - 0.05 * w_classical.grad
		print("torch")


def benchmark_machine_learning(hyperparams={}, num_repeats=1):
	"""Trains a hybrid quantum-classical machine learning pipeline.

	The data is generated from Gaussian blobs. The model first multiplies the input vectors with
	a weight matrix, and then feeds it into a quantum model that uses AngleEmbedding for the encoding
	and BasicEntanglingLayers as the trainable circuit. The number of qubits and layers correspond to the
	number of features. Training uses gradient descent with 50 steps.

	Args:
	hyperparams (dict): hyperparameters to configure this benchmark

		* 'n_features': Number of features of each data sample. Defaults to 4.

		* 'n_samples': Number of data samples to use. Defaults to 20.

		* 'diff_method': name of differentiation method. Defaults to 'best'.

		* 'device': device on which the circuit is run, or valid device name. Defaults to 'default.qubit.

		* 'interface': name of the interface to use. Defaults to 'autograd'.

	num_repeats (int): How often the same circuit is evaluated in a for loop. Default is 1.
	"""

	data, device, diff_method, interface = _ml_defaults(hyperparams)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def quantum_model(x, params):
		qml.templates.AngleEmbedding(x, wires=range(len(x)))
		qml.templates.BasicEntanglerLayers(params, wires=range(len(x)))
		return qml.expval(qml.PauliZ(0))

	for _ in range(num_repeats):

		if interface == 'autograd':
			_machine_learning_autograd(quantum_model, data)

		elif interface == 'tf':
			_machine_learning_tf(quantum_model, data)

		elif interface == 'torch':
			_machine_learning_torch(quantum_model, data)

		# TODO: jax