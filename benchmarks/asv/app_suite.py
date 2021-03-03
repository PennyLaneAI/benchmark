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
Define asv benchmark suite that estimates the speed of applications.
"""

from ..benchmark_functions.vqe import benchmark_vqe
from ..benchmark_functions.hamiltonians import h_LiH
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines import UCCSD
from functools import partial
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import qchem


class VQE:
    """Benchmark the VQE algorithm using different number of optimization steps and grouping
     options."""

    params = ([1, 3], [False, True])
    param_names = ['n_steps', 'optimize']

    def time_hydrogen(self, n_steps, optimize):
        """Time a VQE algorithm with the UCCSD ansatz for computing the ground state energy of the
         hydrogen molecule."""
        hyperparams = {'n_steps': n_steps,
                       'optimize': optimize}
        benchmark_vqe(hyperparams)

    def peakmem_hydrogen(self, n_steps, optimize):
        """Benchmark the peak memory usage of the VQE algorithm with the UCCSD ansatz for computing
         the ground state energy of the hydrogen molecule."""
        hyperparams = {'n_steps': n_steps,
                       'optimize': optimize}
        benchmark_vqe(hyperparams)


class VQE_heavy:
    """Benchmark the VQE algorithm using different grouping options for the lithium hydride molecule
     with 2 active electrons and 8 active spin-orbitals. The sto-3g basis set and UCCSD ansatz are
     used."""

    params = ([False, True])
    param_names = ['optimize']

    def setup(self, optimize):

        electrons = 2
        qubits = 8

        singles, doubles = qchem.excitations(electrons, qubits)
        s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
        hf_state = qchem.hf_state(electrons, qubits)

        self.ham = h_LiH

        self.ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

        self.parameters = np.array([6.39225682, -0.99471664, -4.2026237, -4.48579097, 9.8033157,
                               1.19030864, -3.89924719, 7.25037131, -0.95897967, -0.75287453,
                               0.92252162, 1.10633277, 0.94911997, 1.09138887, 5.27297259])

        self.device = qml.device('default.qubit', wires=qubits)

    def time_lih(self, optimize):
        """Time the VQE algorithm for the lithium hydride molecule."""

        hyperparams = {'ham': self.ham,
                       'ansatz': self.ansatz,
                       'params': self.parameters,
                       'device': self.device,
                       'optimize': optimize}

        benchmark_vqe(hyperparams)

    def peakmem_lih(self, optimize):
        """Benchmark the peak memory usage of the VQE algorithm for the lithium hydride molecule."""

        hyperparams = {'ham': self.ham,
                       'ansatz': self.ansatz,
                       'params': self.parameters,
                       'device': self.device,
                       'optimize': optimize}

        benchmark_vqe(hyperparams)
