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
Benchmarks for the autograd interface.
"""
from math import pi
import numpy as np
import pennylane as qml
qml.enable_tape()


class BenchmarksCore:

	@staticmethod
	def benchmark_circuit_evaluation(
			n_wires=5,
			n_layers=7,
			device=None,
			diff_method=None,
			template=None,
			measurement=None
	):
		"""
		Computes the output of a simple qubit-based quantum circuit.

		Args:
			n_wires (int): number of wires to use
			n_layers (int): number of times the template gets repeated
			device (qml.Device or str): device on which the circuit is run, or valid device name
			diff_method (str): name of differentiation method
			template: Template, like `qml.templates.BasicEntanglerLayers(params, wires=range(5))`. If not None,
				the n_layers parameter will be ignored.
			measurement: measurement function like `qml.expval(qml.PauliZ(0)))`
				...
		"""
		# Set device defaults
		if device is None:
			device = qml.device('default.qubit', wires=n_wires)
		elif isinstance(device, str):
			device = qml.device(device, wires=n_wires)

		# Set diff-method default
		if diff_method is None:
			diff_method = 'parameter-shift'

		# Set template default
		if template is None:
			params = np.random.uniform(high=2 * pi, size=(n_layers, n_wires))
			template = qml.templates.BasicEntanglerLayers(params, wires=range(n_wires))

		# Set measurement default
		if measurement is None:
			measurement = qml.expval(qml.PauliZ(0))

		@qml.qnode(device, diff_method=diff_method)
		def circuit():
			template
			measurement.queue()
			return measurement

		circuit()
