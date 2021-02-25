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
from ..benchmark_functions.qaoa import benchmark_qaoa
import networkx as nx

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

class QAOA:
    """Benchmark the QAOA algorithm using different number of layers."""

    params = ([1, 5])
    param_names = ['n_layers']

    def time_minvertex(self, n_layers):
        """Time a QAOA algorithm for finding the minimum vertex cover of a graph."""
        hyperparams = {'n_layers': n_layers}
        benchmark_qaoa(hyperparams)

    def peakmem_minvertex(self, n_layers):
        """Benchmark the peak memory usage of QAOA algorithm for finding the minimum vertex cover of
        a graph."""
        hyperparams = {'n_layers': n_layers}
        benchmark_qaoa(hyperparams)

class QAOA_heavy:
    """Benchmark the QAOA algorithm for finding the minimum vertex cover of a large graph using
    a large number of layers."""

    n_layers = 5
    graph = nx.complete_graph(20)

    def time_minvertex_heavy(self):
        """Time a QAOA algorithm for finding the minimum vertex cover of a graph."""
        hyperparams = {'n_layers': self.n_layers,
                       'graph': self.graph}
        benchmark_qaoa(hyperparams)

    def peakmem_minvertex_heavy(self):
        """Benchmark the peak memory usage of a QAOA algorithm for finding the minimum vertex cover
        of a graph."""
        hyperparams = {'n_layers': self.n_layers,
                       'graph': self.graph}
        benchmark_qaoa(hyperparams)
