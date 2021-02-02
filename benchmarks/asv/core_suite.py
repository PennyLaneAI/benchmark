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
Define asv benchmark suite that estimates the speed of core operations.
"""
from ..benchmark_functions.circuit import benchmark_circuit
from ..benchmark_functions.gradient import benchmark_gradient
from ..benchmark_functions.optimization import benchmark_optimization


class CircuitEvaluation:
    """Benchmark the evaluation of a circuit using different widths and depths."""

    params = ([2, 5, 10], [3, 6, 9])
    param_names = ['n_wires', 'n_layers']

    def time_circuit(self, n_wires, n_layers):
        """Time a simple default circuit."""
        hyperparams = {'n_wires': n_wires,
                       'n_layers': n_layers}
        benchmark_circuit(hyperparams)


class GradientComputation:
    """Time the computation of a gradient using different widths and depths."""

    params = ([2, 5], [3, 6])
    param_names = ['n_wires', 'n_layers']

    def time_gradient_autograd(self, n_wires, n_layers):
        hyperparams = {'n_wires': n_wires,
                       'n_layers': n_layers,
                       'interface': 'autograd'}
        benchmark_gradient(hyperparams)

    def time_gradient_tf(self, n_wires, n_layers):
        hyperparams = {'n_wires': n_wires,
                       'n_layers': n_layers,
                       'interface': 'tf'}
        benchmark_gradient(hyperparams)

    def time_gradient_torch(self, n_wires, n_layers):
        hyperparams = {'n_wires': n_wires,
                       'n_layers': n_layers,
                       'interface': 'torch'}
        benchmark_gradient(hyperparams)


class Optimization:
    """Benchmark the optimization of a circuit."""

    def time_optimization_autograd(self):
        """Time gradient descent on the default circuit using autograd."""
        hyperparams = {'interface': 'autograd'}
        benchmark_optimization(hyperparams, n_steps=10)

    def time_optimization_tf(self):
        """Time gradient descent on the default circuit using tf."""
        hyperparams = {'interface': 'tf'}
        benchmark_optimization(hyperparams, n_steps=10)

    def time_optimization_torch(self):
        """Time gradient descent on the default circuit using torch."""
        hyperparams = {'interface': 'torch'}
        benchmark_optimization(hyperparams, n_steps=10)
