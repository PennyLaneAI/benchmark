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

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines import UCCSD
from functools import partial
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import qchem
from ..benchmark_functions.vqe import benchmark_vqe
from ..benchmark_functions.qaoa import benchmark_qaoa
import networkx as nx


class VQE_light:
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

        H_coeffs = np.array([-6.74845266e+00, -1.02553930e-01, 1.00530907e-02, 1.00530907e-02,
                             -1.02553930e-01, 1.00530907e-02, 1.00530907e-02, -2.76355319e-01,
                             -2.76355319e-01, -2.96925596e-01, -2.96925596e-01, -2.96925596e-01,
                             -2.96925596e-01, 1.21916192e-01, 1.21233148e-02, 1.21233148e-02,
                             1.21233148e-02, 1.21233148e-02, 3.25324294e-03, -3.25324294e-03,
                             -3.25324294e-03, 3.25324294e-03, 5.86266678e-03, -5.86266678e-03,
                             -5.86266678e-03, 5.86266678e-03, 5.86266678e-03, -5.86266678e-03,
                             -5.86266678e-03, 5.86266678e-03, 5.26857432e-02, 5.59389862e-02,
                             -1.85422006e-03, -1.85422006e-03, 4.81813200e-03, -4.81813200e-03,
                             -4.81813200e-03, 4.81813200e-03, 4.81813200e-03, -4.81813200e-03,
                             -4.81813200e-03, 4.81813200e-03, 6.17431075e-02, 3.39017831e-03,
                             3.39017831e-03, 6.76057742e-02, -1.42795369e-03, -1.42795369e-03,
                             6.17431075e-02, 3.39017831e-03, 3.39017831e-03, 6.76057742e-02,
                             -1.42795369e-03, -1.42795369e-03, 5.59389862e-02, -1.85422006e-03,
                             -1.85422006e-03, -4.81813200e-03, -4.81813200e-03, -4.81813200e-03,
                             -4.81813200e-03, -4.81813200e-03, -4.81813200e-03, -4.81813200e-03,
                             -4.81813200e-03, 5.26857432e-02, 6.76057742e-02, -1.42795369e-03,
                             -1.42795369e-03, 6.17431075e-02, 3.39017831e-03, 3.39017831e-03,
                             6.76057742e-02, -1.42795369e-03, -1.42795369e-03, 6.17431075e-02,
                             3.39017831e-03, 3.39017831e-03, 8.44840116e-02, 1.03194543e-02,
                             -1.03194543e-02, -1.03194543e-02, 1.03194543e-02, 1.03194543e-02,
                             -1.03194543e-02, -1.03194543e-02, 1.03194543e-02, 6.01815510e-02,
                             7.05010052e-02, 6.01815510e-02, 7.05010052e-02, 7.05010052e-02,
                             6.01815510e-02, 7.05010052e-02, 6.01815510e-02, 7.82363778e-02,
                             4.21728488e-03, -4.21728488e-03, -4.21728488e-03, 4.21728488e-03,
                             6.55845232e-02, 6.98018080e-02, 6.98018080e-02, 6.55845232e-02,
                             7.82363778e-02])

        H_ops = [Identity(wires=[0]),
             PauliZ(wires=[0]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]),
             PauliZ(wires=[1]),
             PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]),
             PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]),
             PauliZ(wires=[2]),
             PauliZ(wires=[3]),
             PauliZ(wires=[4]),
             PauliZ(wires=[5]),
             PauliZ(wires=[6]),
             PauliZ(wires=[7]),
             PauliZ(wires=[0]) @ PauliZ(wires=[1]),
             PauliY(wires=[0]) @ PauliY(wires=[2]),
             PauliX(wires=[0]) @ PauliX(wires=[2]),
             PauliZ(wires=[0]) @ PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]),
             PauliZ(wires=[0]) @ PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]),
             PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
             PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
             PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
             PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
             PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
             PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
             PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
             PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
             PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[6]) @ PauliY(wires=[7]),
             PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[6]) @ PauliX(wires=[7]),
             PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[6]) @ PauliY(wires=[7]),
             PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[6]) @ PauliX(wires=[7]),
             PauliZ(wires=[0]) @ PauliZ(wires=[2]),
             PauliZ(wires=[0]) @ PauliZ(wires=[3]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[3]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[3]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliX(
                 wires=[4]) @ PauliY(wires=[5]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliX(
                 wires=[4]) @ PauliX(wires=[5]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliY(
                 wires=[4]) @ PauliY(wires=[5]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliY(
                 wires=[4]) @ PauliX(wires=[5]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliX(
                 wires=[6]) @ PauliY(wires=[7]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliX(
                 wires=[6]) @ PauliX(wires=[7]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliY(
                 wires=[6]) @ PauliY(wires=[7]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliY(
                 wires=[6]) @ PauliX(wires=[7]),
             PauliZ(wires=[0]) @ PauliZ(wires=[4]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[4]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[4]),
             PauliZ(wires=[0]) @ PauliZ(wires=[5]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[5]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[5]),
             PauliZ(wires=[0]) @ PauliZ(wires=[6]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[6]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[6]),
             PauliZ(wires=[0]) @ PauliZ(wires=[7]),
             PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[7]),
             PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[7]),
             PauliZ(wires=[1]) @ PauliZ(wires=[2]),
             PauliY(wires=[1]) @ PauliY(wires=[3]),
             PauliX(wires=[1]) @ PauliX(wires=[3]),
             PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
             PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
             PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
             PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
             PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[6]) @ PauliY(wires=[7]),
             PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[6]) @ PauliY(wires=[7]),
             PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[6]) @ PauliX(wires=[7]),
             PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[6]) @ PauliX(wires=[7]),
             PauliZ(wires=[1]) @ PauliZ(wires=[3]),
             PauliZ(wires=[1]) @ PauliZ(wires=[4]),
             PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[4]),
             PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[4]),
             PauliZ(wires=[1]) @ PauliZ(wires=[5]),
             PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[5]),
             PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[5]),
             PauliZ(wires=[1]) @ PauliZ(wires=[6]),
             PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[6]),
             PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[6]),
             PauliZ(wires=[1]) @ PauliZ(wires=[7]),
             PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[7]),
             PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[7]),
             PauliZ(wires=[2]) @ PauliZ(wires=[3]),
             PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
             PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
             PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
             PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
             PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[6]) @ PauliY(wires=[7]),
             PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[6]) @ PauliX(wires=[7]),
             PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[6]) @ PauliY(wires=[7]),
             PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[6]) @ PauliX(wires=[7]),
             PauliZ(wires=[2]) @ PauliZ(wires=[4]),
             PauliZ(wires=[2]) @ PauliZ(wires=[5]),
             PauliZ(wires=[2]) @ PauliZ(wires=[6]),
             PauliZ(wires=[2]) @ PauliZ(wires=[7]),
             PauliZ(wires=[3]) @ PauliZ(wires=[4]),
             PauliZ(wires=[3]) @ PauliZ(wires=[5]),
             PauliZ(wires=[3]) @ PauliZ(wires=[6]),
             PauliZ(wires=[3]) @ PauliZ(wires=[7]),
             PauliZ(wires=[4]) @ PauliZ(wires=[5]),
             PauliY(wires=[4]) @ PauliX(wires=[5]) @ PauliX(wires=[6]) @ PauliY(wires=[7]),
             PauliY(wires=[4]) @ PauliY(wires=[5]) @ PauliX(wires=[6]) @ PauliX(wires=[7]),
             PauliX(wires=[4]) @ PauliX(wires=[5]) @ PauliY(wires=[6]) @ PauliY(wires=[7]),
             PauliX(wires=[4]) @ PauliY(wires=[5]) @ PauliY(wires=[6]) @ PauliX(wires=[7]),
             PauliZ(wires=[4]) @ PauliZ(wires=[6]),
             PauliZ(wires=[4]) @ PauliZ(wires=[7]),
             PauliZ(wires=[5]) @ PauliZ(wires=[6]),
             PauliZ(wires=[5]) @ PauliZ(wires=[7]),
             PauliZ(wires=[6]) @ PauliZ(wires=[7])]

        electrons = 2
        qubits = 8

        singles, doubles = qchem.excitations(electrons, qubits)
        s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
        hf_state = qchem.hf_state(electrons, qubits)

        self.ham = qml.Hamiltonian(H_coeffs, H_ops)

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


class QAOA_light:
    """Benchmark the QAOA algorithm for finding the minimum vertex cover of a small graph using
    different number of layers."""

    params = ([1, 5])
    param_names = ['n_layers']

    def time_minvertex_light(self, n_layers):
        """Time a QAOA algorithm for finding the minimum vertex cover of a small graph."""
        hyperparams = {'n_layers': n_layers}
        benchmark_qaoa(hyperparams)

    def peakmem_minvertex_light(self, n_layers):
        """Benchmark the peak memory usage of QAOA algorithm for finding the minimum vertex cover of
        a small graph."""
        hyperparams = {'n_layers': n_layers}
        benchmark_qaoa(hyperparams)

class QAOA_heavy:
    """Benchmark the QAOA algorithm for finding the minimum vertex cover of a large graph."""

    n_layers = 5
    graph = nx.complete_graph(20)

    def time_minvertex_heavy(self):
        """Time a QAOA algorithm for finding the minimum vertex cover of a large graph."""
        hyperparams = {'n_layers': self.n_layers,
                       'graph': self.graph}
        benchmark_qaoa(hyperparams)

    def peakmem_minvertex_heavy(self):
        """Benchmark the peak memory usage of a QAOA algorithm for finding the minimum vertex cover
        of a large graph."""
        hyperparams = {'n_layers': self.n_layers,
                       'graph': self.graph}
        benchmark_qaoa(hyperparams)
