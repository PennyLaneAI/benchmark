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

def benchmark_casual(dev_name, s3=None):
    """ A simple optimization workflow

    Args:
        dev_name (str): Either "local", "sv1", "tn1", or "ionq"
        s3 (tuple):  A tuple of (bucket, prefix) to specify the s3 storage location

    """
    n_steps = 2
    n_wires = 4
    n_layers = 6
    interface = 'autograd'
    diff_method = 'best'

    if dev_name == "local":
        device = qml.device("braket.local.qubit", wires=n_wires, shots=None)
    elif dev_name == "sv1":
        device = qml.device("braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_destination_folder=s3,
            wires=n_wires,
            shots=None)
    elif dev_name == "tn1":
        shots = 1000
        device = qml.device("braket.aws.qubit",
                            device_arn='arn:aws:braket:::device/quantum-simulator/amazon/tn1',
                            s3_destination_folder=s3,
                            wires=n_wires,
                            shots=shots)
    elif dev_name == "ionq":
        shots = 1000
        device = qml.device("braket.aws.qubit",
                    device_arn='arn:aws:braket:::device/qpu/ionq/ionQdevice',
                    s3_destination_folder=s3,
                    wires=n_wires,
                    shots=shots)
    else:
        raise ValueError("dev_name not 'local', 'sv1', 'ionq', or 'tn1'")

    @qml.qnode(device, interface=interface, diff_method=diff_method)
	def circuit(params_):
		qml.templates.BasicEntanglerLayers(params_, wires=range(n_wires))
		return qml.expval(qml.PauliZ(0))

    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    print("starting optimization")
    for i in range(n_steps):
        params = opt.step(circuit, params)
        print("step: ", i)
    
def benchmark_power(dev_name, s3=None)
    """ A substantial QAOA workflow

    Args:
        dev_name (str): Either "local", "sv1", "tn1", or "ionq"
        s3 (tuple):  A tuple of (bucket, prefix) to specify the s3 storage location
    """
    n_layers = 5
    if dev_name == "ionq":
        graph = nx.complete_graph(11)
    else:
    	graph = nx.complete_graph(20)
        
	params = [[0.5] * n_layers, [0.5] * n_layers]
	interface = 'autograd'
	diff_method = 'best'
	n_wires = len(graph.nodes)
	H_cost, H_mixer = qaoa.min_vertex_cover(graph, constrained=False)

    if dev_name == "local":
        device = qml.device("braket.local.qubit", wires=n_wires, shots=None)
    elif dev_name == "sv1":
        device = qml.device("braket.aws.qubit",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_destination_folder=s3,
            wires=n_wires,
            shots=None)
    elif dev_name == "tn1":
        shots=1000
        device = qml.device("braket.aws.qubit",
                            device_arn='arn:aws:braket:::device/quantum-simulator/amazon/tn1',
                            s3_destination_folder=s3,
                            wires=n_wires,
                            shots=shots)
    elif dev_name == "ionq":
        shots = 1000
        device = qml.device("braket.aws.qubit",
                    device_arn='arn:aws:braket:::device/qpu/ionq/ionQdevice',
                    s3_destination_folder=s3,
                    wires=n_wires,
                    shots=shots)
    else:
        raise ValueError("dev_name not 'local', 'sv1','tn1', or 'ionq'")

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