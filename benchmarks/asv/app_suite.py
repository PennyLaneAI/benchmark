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
Define asv benchmark suite that estimates the speed of applications.
"""

import pennylane as qml
from pennylane import numpy as np
from ..benchmark_functions.vqe import benchmark_vqe
from pennylane import Identity, PauliX, PauliY, PauliZ

from pennylane import qchem
from pennylane.templates.subroutines import UCCSD
from functools import partial


class VQEEvaluation:
    """Benchmark VQE."""

    params = ([1, 2], [UCCSD])
    param_names = ['n_steps', 'template']

    def time_hydrogen(self, n_steps, template):
        """Time VQE for the hydrogen molecule with sto-3g sis set."""

        H_coeffs = np.array([-0.05963862, 0.17575739, 0.17575739, -0.23666489, -0.23666489,
                             0.17001485, 0.04491735, -0.04491735, -0.04491735, 0.04491735,
                             0.12222641, 0.16714376, 0.16714376, 0.12222641, 0.17570278])

        H_ops = [Identity(wires=[0]),
                 PauliZ(wires=[0]),
                 PauliZ(wires=[1]),
                 PauliZ(wires=[2]),
                 PauliZ(wires=[3]),
                 PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                 PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                 PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                 PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                 PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                 PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                 PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                 PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                 PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                 PauliZ(wires=[2]) @ PauliZ(wires=[3])]

        H = qml.Hamiltonian(H_coeffs, H_ops)

        electrons = 2
        qubits = 4

        singles, doubles = qchem.excitations(electrons, qubits, delta_sz=0)
        s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
        hf_state = qchem.hf_state(electrons, qubits)
        ansatz = partial(template, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

        params = np.random.normal(0, np.pi, len(singles) + len(doubles))

        dev = qml.device('default.qubit', wires=qubits)

        benchmark_vqe(H, ansatz, dev, params, n_steps)
