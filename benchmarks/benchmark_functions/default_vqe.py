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
Benchmarks for VQE simulation.
"""
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines import UCCSD
from functools import partial
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import qchem


def _set_defaults(hyperparams):
	"""Uses hyperparameters or defaults to construct the components of the VQE circuit for the
	hydrogen molecule with the sto-3g basis set.

	Args:
		hyperparams (dict): hyperparameters provided by user
	"""

	H_coeffs = np.array([-0.05963862, 0.17575739, 0.17575739, -0.23666489, -0.23666489,
						  0.17001485, 0.04491735, -0.04491735, -0.04491735, 0.04491735,
						  0.12222641, 0.16714376, 0.16714376, 0.12222641, 0.17570278])

	H_ops = [Identity(wires=[0]), PauliZ(wires=[0]), PauliZ(wires=[1]), PauliZ(wires=[2]),
			 PauliZ(wires=[3]), PauliZ(wires=[0]) @ PauliZ(wires=[1]),
			 PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
			 PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
			 PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
			 PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
			 PauliZ(wires=[0]) @ PauliZ(wires=[2]), PauliZ(wires=[0]) @ PauliZ(wires=[3]),
			 PauliZ(wires=[1]) @ PauliZ(wires=[2]), PauliZ(wires=[1]) @ PauliZ(wires=[3]),
			 PauliZ(wires=[2]) @ PauliZ(wires=[3])]

	electrons = 2
	qubits = 4

	singles, doubles = qchem.excitations(2, 4, delta_sz=0)
	s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
	hf_state = qchem.hf_state(electrons, qubits)
	ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

	Hamiltonian = hyperparams.pop('Hamiltonian', qml.Hamiltonian(H_coeffs, H_ops))
	ansatz = hyperparams.pop('template', ansatz)
	params = hyperparams.pop('params', np.random.normal(0, np.pi, len(singles) + len(doubles)))
	n_steps = hyperparams.pop('n_steps', 1)
	device = hyperparams.pop('device', 'default.qubit')
	interface = hyperparams.pop('interface', 'autograd')
	diff_method = hyperparams.pop('diff_method', 'best')
	optimize = hyperparams.pop('optimize', True)

	# if device name is given, create device
	if isinstance(device, str):
		device = qml.device(device, wires=qubits)

	return Hamiltonian, ansatz, params, n_steps, device, interface, diff_method, optimize
