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
Define asv benchmark suite that estimates the speed of different devices.
"""
from ..benchmark_functions.circuit import benchmark_circuit

# List of devices to test.
# The benchmark will fail if a device is not installed.
DEVICES = ['default.qubit',
           'lightning.qubit',
           'default.mixed',
           'qiskit.aer',
           'qiskit.basicaer',
           'cirq.simulator',
           'cirq.pasqal',
           'cirq.qsim',
           'qulacs.simulator'
           ]


class CircuitEvaluation:
    """Benchmark the evaluation of a circuit using different widths and depths."""

    params = (DEVICES,
              [2, 5, 10],
              [3, 6, 9])
    param_names = ['device',
                   'n_wires',
                   'n_layers']

    def time_circuit(self, dev, n_wires, n_layers):
        """Time a simple default circuit."""
        hyperparams = {'n_wires': n_wires,
                       'n_layers': n_layers,
                       'device': dev}
        benchmark_circuit(hyperparams)
