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
Benchmarks for VQE simulations.
"""
import pennylane as qml
from .default_vqe import _set_defaults

qml.enable_tape()


def benchmark_vqe(hyperparams={}):
	"""
	Performs VQE optimizations.

	Args:
		hyperparams (dict): hyperparameters to configure this benchmark

			* 'Hamiltonian': Molecular Hamiltonian represented as a PennyLane Hamiltonian class

			* 'ansatz': VQE ansatz

			* 'params': Numpy array of trainable parameters that is fed into the ansatz.

			* 'n_steps': Number of VQE steps

			* 'device': Device on which the circuit is run, or valid device name

			* 'interface': Name of the interface to use

			* 'diff_method': Name of differentiation method

			* 'optimize': argument for grouping the observables composing the Hamiltonian
	"""

	Hamiltonian, ansatz, params, n_steps, device, interface, diff_method, optimize = _set_defaults(hyperparams)

	cost_fn = qml.ExpvalCost(ansatz, Hamiltonian, device, interface, diff_method, optimize)

	opt = qml.GradientDescentOptimizer(stepsize=0.4)

	for n in range(n_steps):
		params = opt.step(cost_fn, params)
		energy = cost_fn(params)
