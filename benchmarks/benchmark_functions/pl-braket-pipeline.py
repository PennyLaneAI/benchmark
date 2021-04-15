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
Fixed benchmarks for pennylane-braket pipelines.
"""
from numpy.random import random
import pennylane as qml
import networkx as nx
from pennylane import numpy as pnp
from pennylane import qaoa


def benchmark_casual_sv1(bucket, prefix):
	"""Trains a small quantum circuit for 20 steps using the remote Braket SV1 simulator.
	The entire training is repeated 5 times.
	"""
	num_repeats = 5
	n_steps = 20
	n_wires = 4
	n_layers = 6
	interface = 'autograd'
	params = random(size=(n_layers, n_wires))
	diff_method = 'best'
	s3 = (bucket, prefix)
	device = qml.device("braket.aws.qubit",
						device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
						s3_destination_folder=s3,
						wires=n_wires,
						shots=None)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params_):
		qml.templates.BasicEntanglerLayers(params_, wires=range(n_wires))
		return qml.expval(qml.PauliZ(0))

	for _ in range(num_repeats):

		params = pnp.array(params, requires_grad=True)
		opt = qml.GradientDescentOptimizer(stepsize=0.1)

		for i in range(n_steps):
			params = opt.step(circuit, params)


def benchmark_power_sv1(bucket, prefix):
	"""
	Performs a QAOA optimization on 20 qubits with 5 layers on the remote Braket SV1 simulator.
	"""

	n_layers = 5
	graph = nx.complete_graph(20)
	params = [[0.5] * n_layers, [0.5] * n_layers]
	interface = 'autograd'
	diff_method = 'best'
	n_wires = len(graph.nodes)
	H_cost, H_mixer = qaoa.min_vertex_cover(graph, constrained=False)

	s3 = (bucket, prefix)
	device = qml.device("braket.aws.qubit",
						device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
						s3_destination_folder=s3,
						wires=n_wires,
						shots=None)

	def qaoa_layer(gamma, alpha):
		qaoa.cost_layer(gamma, H_cost)
		qaoa.mixer_layer(alpha, H_mixer)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params):
		for w in range(n_wires):
			qml.Hadamard(wires=w)
		qml.layer(qaoa_layer, n_layers, params[0], params[1])
		return [qml.sample(qml.PauliZ(i)) for i in range(n_wires)]

	circuit(params)


def benchmark_casual_tn1(bucket, prefix):
	"""Trains a small quantum circuit for 20 steps using the remote Braket TN1 simulator.
	The entire training is repeated 5 times.
	"""
	num_repeats = 5
	n_steps = 20
	n_wires = 4
	n_layers = 6
	interface = 'autograd'
	params = random(size=(n_layers, n_wires))
	diff_method = 'best'
	shots = 1000

	s3 = (bucket, prefix)
	device = qml.device("braket.aws.qubit",
						device_arn='arn:aws:braket:::device/quantum-simulator/amazon/tn1',
						s3_destination_folder=s3,
						wires=n_wires,
						shots=shots)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params_):
		qml.templates.BasicEntanglerLayers(params_, wires=range(n_wires))
		return qml.expval(qml.PauliZ(0))

	for _ in range(num_repeats):

		params = pnp.array(params, requires_grad=True)
		opt = qml.GradientDescentOptimizer(stepsize=0.1)

		for i in range(n_steps):
			params = opt.step(circuit, params)


def benchmark_power_tn1(bucket, prefix):
	"""
	Performs a QAOA optimization on 20 qubits with 5 layers on the remote Braket TN1 simulator.
	"""

	n_layers = 5
	graph = nx.complete_graph(20)
	params = [[0.5] * n_layers, [0.5] * n_layers]
	interface = 'autograd'
	diff_method = 'best'
	n_wires = len(graph.nodes)
	H_cost, H_mixer = qaoa.min_vertex_cover(graph, constrained=False)
	shots = 1000

	s3 = (bucket, prefix)
	device = qml.device("braket.aws.qubit",
						device_arn='arn:aws:braket:::device/quantum-simulator/amazon/tn1',
						s3_destination_folder=s3,
						wires=n_wires,
						shots=shots)

	def qaoa_layer(gamma, alpha):
		qaoa.cost_layer(gamma, H_cost)
		qaoa.mixer_layer(alpha, H_mixer)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params):
		for w in range(n_wires):
			qml.Hadamard(wires=w)
		qml.layer(qaoa_layer, n_layers, params[0], params[1])
		return [qml.sample(qml.PauliZ(i)) for i in range(n_wires)]

	circuit(params)


def benchmark_casual_ionq(bucket, prefix):
	"""Trains a small quantum circuit for 20 steps using the remote Braket IonQ hardware.
	The entire training is repeated 5 times.
	"""
	num_repeats = 5
	n_steps = 20
	n_wires = 4
	n_layers = 6
	interface = 'autograd'
	params = random(size=(n_layers, n_wires))
	diff_method = 'best'
	shots = 1000

	s3 = (bucket, prefix)
	device = qml.device("braket.aws.qubit",
						device_arn='arn:aws:braket:::device/qpu/ionq/ionQdevice',
						s3_destination_folder=s3,
						wires=n_wires,
						shots=shots)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params_):
		qml.templates.BasicEntanglerLayers(params_, wires=range(n_wires))
		return qml.expval(qml.PauliZ(0))

	for _ in range(num_repeats):

		params = pnp.array(params, requires_grad=True)
		opt = qml.GradientDescentOptimizer(stepsize=0.1)

		for i in range(n_steps):
			params = opt.step(circuit, params)


def benchmark_power_ionq(bucket, prefix):
	"""
	Performs a QAOA optimization on 20 qubits with 5 layers on the remote Braket IonQ hardware.
	"""

	n_layers = 5
	graph = nx.complete_graph(20)
	params = [[0.5] * n_layers, [0.5] * n_layers]
	interface = 'autograd'
	diff_method = 'best'
	n_wires = len(graph.nodes)
	H_cost, H_mixer = qaoa.min_vertex_cover(graph, constrained=False)
	shots = 1000

	s3 = (bucket, prefix)
	device = qml.device("braket.aws.qubit",
						device_arn='arn:aws:braket:::device/qpu/ionq/ionQdevice',
						s3_destination_folder=s3,
						wires=n_wires,
						shots=shots)

	def qaoa_layer(gamma, alpha):
		qaoa.cost_layer(gamma, H_cost)
		qaoa.mixer_layer(alpha, H_mixer)

	@qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params):
		for w in range(n_wires):
			qml.Hadamard(wires=w)
		qml.layer(qaoa_layer, n_layers, params[0], params[1])
		return [qml.sample(qml.PauliZ(i)) for i in range(n_wires)]

	circuit(params)
