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
from ..benchmark_functions.vqe import benchmark_vqe
from ..benchmark_functions.hamiltonians import ham_lih
from ..benchmark_functions.qaoa import benchmark_qaoa
from ..benchmark_functions.machine_learning import benchmark_machine_learning

import networkx as nx


class VQE_light:
    """Benchmark the VQE algorithm using different number of optimization steps and grouping
    options."""

    params = ([1, 3], [False, True])
    param_names = ["n_steps", "optimize"]

    def time_hydrogen(self, n_steps, optimize):
        """Time a VQE algorithm with the UCCSD ansatz for computing the ground state energy of the
        hydrogen molecule."""
        hyperparams = {"n_steps": n_steps, "optimize": optimize}
        benchmark_vqe(hyperparams)

    def peakmem_hydrogen(self, n_steps, optimize):
        """Benchmark the peak memory usage of the VQE algorithm with the UCCSD ansatz for computing
        the ground state energy of the hydrogen molecule."""
        hyperparams = {"n_steps": n_steps, "optimize": optimize}
        benchmark_vqe(hyperparams)


class VQE_heavy:
    """Benchmark the VQE algorithm using different grouping options for the lithium hydride molecule
    with 2 active electrons and 8 active spin-orbitals. The sto-3g basis set and UCCSD ansatz are
    used."""

    params = [False, True]
    param_names = ["optimize"]

    timeout = 600  # 10 minutes
    repeat = (1, 1, 600)  # Only collect one sample
    number = 1  # one iteration in each sample

    def setup(self, optimize):

        qubits = 8

        s_wires = [[0, 1, 2],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4, 5, 6],
                    [1, 2, 3],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6, 7]]

        d_wires = [[[0, 1], [2, 3]],
                    [[0, 1], [2, 3, 4, 5]],
                    [[0, 1], [2, 3, 4, 5, 6, 7]],
                    [[0, 1], [3, 4]],
                    [[0, 1], [3, 4, 5, 6]],
                    [[0, 1], [4, 5]],
                    [[0, 1], [4, 5, 6, 7]],
                    [[0, 1], [5, 6]],
                    [[0, 1], [6, 7]]]

        hf_state = np.array([1, 1, 0, 0, 0, 0, 0, 0])

        self.ham = ham_lih

        self.ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

        self.parameters = np.array(
            [
                6.39225682,
                -0.99471664,
                -4.2026237,
                -4.48579097,
                9.8033157,
                1.19030864,
                -3.89924719,
                7.25037131,
                -0.95897967,
                -0.75287453,
                0.92252162,
                1.10633277,
                0.94911997,
                1.09138887,
                5.27297259,
            ]
        )

        self.device = qml.device("default.qubit", wires=qubits)

    def time_lih(self, optimize):
        """Time the VQE algorithm for the lithium hydride molecule."""

        hyperparams = {
            "ham": self.ham,
            "ansatz": self.ansatz,
            "params": self.parameters,
            "device": self.device,
            "optimize": optimize,
        }

        benchmark_vqe(hyperparams)

    def peakmem_lih(self, optimize):
        """Benchmark the peak memory usage of the VQE algorithm for the lithium hydride molecule."""

        hyperparams = {
            "ham": self.ham,
            "ansatz": self.ansatz,
            "params": self.parameters,
            "device": self.device,
            "optimize": optimize,
        }

        benchmark_vqe(hyperparams)


class QAOA_light:
    """Benchmark the QAOA algorithm for finding the minimum vertex cover of a small graph using
    different number of layers."""

    params = [1, 5]
    param_names = ["n_layers"]

    def time_minvertex_light(self, n_layers):
        """Time a QAOA algorithm for finding the minimum vertex cover of a small graph."""
        hyperparams = {"n_layers": n_layers}
        benchmark_qaoa(hyperparams)

    def peakmem_minvertex_light(self, n_layers):
        """Benchmark the peak memory usage of QAOA algorithm for finding the minimum vertex cover of
        a small graph."""
        hyperparams = {"n_layers": n_layers}
        benchmark_qaoa(hyperparams)


class QAOA_heavy:
    """Benchmark the QAOA algorithm for finding the minimum vertex cover of a large graph."""

    n_layers = 5
    graph = nx.complete_graph(20)
    timeout = 600  # 10 minutes
    repeat = (1, 1, 600)  # Only collect one sample
    number = 1  # one iteration in each sample

    def time_minvertex_heavy(self):
        """Time a QAOA algorithm for finding the minimum vertex cover of a large graph."""
        hyperparams = {"n_layers": self.n_layers, "graph": self.graph}
        benchmark_qaoa(hyperparams)

    def peakmem_minvertex_heavy(self):
        """Benchmark the peak memory usage of a QAOA algorithm for finding the minimum vertex cover
        of a large graph."""
        hyperparams = {"n_layers": self.n_layers, "graph": self.graph}
        benchmark_qaoa(hyperparams)


class ML_light:
    """Benchmark a hybrid quantum-classical machine learning application with a small dataset."""

    params = ["autograd", "torch", "tf"]
    param_names = ["interface"]
    n_features = 4
    n_samples = 20

    def time_ml_light(self, interface):
        """Time 50 training steps of a hybrid quantum machine learning example."""
        hyperparams = {
            "n_layers": self.n_features,
            "n_samples": self.n_samples,
            "interface": interface,
        }
        benchmark_machine_learning(hyperparams)

    def peakmem_ml_light(self, interface):
        """Benchmark peak memory of 50 training steps of a hybrid quantum machine learning example
        ."""
        hyperparams = {
            "n_layers": self.n_features,
            "n_samples": self.n_samples,
            "interface": interface,
        }
        benchmark_machine_learning(hyperparams)


class ML_heavy:
    """Benchmark a hybrid quantum-classical machine learning application with a large dataset."""

    params = ["autograd", "torch", "tf"]
    param_names = ["interface"]
    n_features = 10
    n_samples = 100

    timeout = 600  # 10 minutes
    repeat = (1, 1, 600)  # Only collect one sample
    number = 1  # one iteration in each sample

    def time_ml_heavy(self, interface):
        """Time 50 training steps of a hybrid quantum machine learning example."""
        hyperparams = {
            "n_layers": self.n_features,
            "n_samples": self.n_samples,
            "interface": interface,
        }
        benchmark_machine_learning(hyperparams)

    def peakmem_ml_heavy(self, interface):
        """Benchmark peak memory of 50 training steps of a hybrid quantum machine learning example."""
        hyperparams = {
            "n_layers": self.n_features,
            "n_samples": self.n_samples,
            "interface": interface,
        }
        benchmark_machine_learning(hyperparams)
