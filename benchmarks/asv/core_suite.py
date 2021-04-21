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


class CircuitEvaluation_light:
    """Benchmark the evaluation of a circuit using different widths and depths."""

    params = ([2, 5, 10], [3, 6, 9])
    param_names = ["n_wires", "n_layers"]

    def time_circuit(self, n_wires, n_layers):
        """Time a simple default circuit."""
        hyperparams = {"n_wires": n_wires, "n_layers": n_layers}
        benchmark_circuit(hyperparams)


class GradientComputation_light:
    """Time the computation of a gradient using different widths and depths."""

    params = ([2, 5], [3, 6], ["autograd", "tf", "torch"])
    param_names = ["n_wires", "n_layers", "interface"]

    def time_gradient(self, n_wires, n_layers, interface):
        hyperparams = {"n_wires": n_wires, "n_layers": n_layers, "interface": interface}
        benchmark_gradient(hyperparams)


class Optimization_light:
    """Benchmark the optimization of a circuit."""

    params = ["autograd", "tf", "torch"]
    param_names = ["interface"]

    def time_optimization(self, interface):
        """Time gradient descent on the default circuit using an interface."""
        hyperparams = {"interface": interface}
        benchmark_optimization(hyperparams, n_steps=10)
