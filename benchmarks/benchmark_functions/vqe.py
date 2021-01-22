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
Benchmarks for VQE simulations.
"""
import pennylane as qml
from pennylane import numpy as np

from pennylane import Identity, PauliX, PauliY, PauliZ

from pennylane import qchem
from pennylane.templates.subroutines import UCCSD
from functools import partial

qml.enable_tape()


def benchmark_vqe(H={}, ansatz={}, dev={}, params={}, n_steps={}):
	"""
	Performs VQE simulation.

	Args:
		H: Hamiltonian.
		ansatz: ansatz to use.
		dev: device on which the circuit is run.
		params: numpy array of trainable parameters.
		n_steps: number of VQE steps.
	"""

	cost_fn = qml.ExpvalCost(ansatz, H, dev, optimize=True)
	opt = qml.GradientDescentOptimizer(stepsize=0.4)

	for n in range(n_steps):
		params = opt.step(cost_fn, params)
		energy = cost_fn(params)

# def benchmark_vqe(hyperparams={}, num_repeats={}):
# 	"""
# 	Performs VQE simulation for the hydrogen molecule with STO-3G basis set.
#
# 	Args:
# 		hyperparams (dict): hyperparameters to configure this benchmark
#
# 			* 'n_wires': Number of wires to use. Will be ignored if custom device and template are provided.
#
# 			* 'n_layers': Number of layers in the default template. Will be ignored if custom params are provided.
#
# 			* 'diff_method': name of differentiation method
#
# 			* 'device': device on which the circuit is run, or valid device name
#
# 			* 'interface': name of the interface to use
#
# 			* 'template': Template to use. The template must take the trainable parameters as its only argument.
#
# 			* 'params': Numpy array of trainable parameters that is fed into the template.
#
# 			* 'measurement': measurement function like `qml.expval(qml.PauliZ(0)))`
#
# 		num_repeats (int): How often the same circuit is evaluated in a for loop. Default is 1.
# 	"""
#
# 	H_coeffs = np.array([-0.05963862, 0.17575739, 0.17575739, -0.23666489, -0.23666489,
# 						 0.17001485, 0.04491735, -0.04491735, -0.04491735, 0.04491735,
# 						 0.12222641, 0.16714376, 0.16714376, 0.12222641, 0.17570278])
#
# 	H_ops = [Identity(wires=[0]),
# 			 PauliZ(wires=[0]),
# 			 PauliZ(wires=[1]),
# 			 PauliZ(wires=[2]),
# 			 PauliZ(wires=[3]),
# 			 PauliZ(wires=[0]) @ PauliZ(wires=[1]),
# 			 PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
# 			 PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
# 			 PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
# 			 PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
# 			 PauliZ(wires=[0]) @ PauliZ(wires=[2]),
# 			 PauliZ(wires=[0]) @ PauliZ(wires=[3]),
# 			 PauliZ(wires=[1]) @ PauliZ(wires=[2]),
# 			 PauliZ(wires=[1]) @ PauliZ(wires=[3]),
# 			 PauliZ(wires=[2]) @ PauliZ(wires=[3])]
#
# 	H = qml.Hamiltonian(H_coeffs, H_ops)
#
# 	electrons = 2
# 	qubits = 4
#
# 	singles, doubles = qchem.excitations(electrons, qubits, delta_sz=0)
# 	s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
# 	hf_state = qchem.hf_state(electrons, qubits)
# 	ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)
#
# 	params = np.random.normal(0, np.pi, len(singles) + len(doubles))
#
# 	dev = qml.device('default.qubit', wires=qubits)
#
# 	cost_fn = qml.ExpvalCost(ansatz, H, dev, optimize=True)
# 	opt = qml.GradientDescentOptimizer(stepsize=0.4)
#
# 	for n in range(num_repeats):
# 		params = opt.step(cost_fn, params)
# 		energy = cost_fn(params)
