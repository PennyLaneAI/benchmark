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
Benchmarks for a machine learning application.
"""
import pennylane as qml
from numpy.random import random
from pennylane.templates import BasicEntanglerLayers
from pennylane.templates.decorator import template as template_decorator


def _set_defaults(hyperparams):
	"""Uses hyperparameters or defaults to construct the components of the circuit.

	Args:
		hyperparams (dict): hyperparameters provided by user
	"""
	# get hyperparameters or set default values
	n_wires = hyperparams.pop('n_wires', 4)
	n_layers = hyperparams.pop('n_layers', 6)
	interface = hyperparams.pop('interface', 'autograd')
	params = hyperparams.pop('params', random(size=(n_layers, n_wires)))
	measurement = hyperparams.pop('measurement', qml.expval(qml.PauliZ(0)))
	diff_method = hyperparams.pop('diff_method', 'parameter-shift')
	device = hyperparams.pop('device', 'default.qubit')
	template = hyperparams.pop('template', None)

	# if device name is given, create device
	if isinstance(device, str):
		device = qml.device(device, wires=n_wires)

	# wrap default template so it only takes the parameters as argument
	if template is None:
		@template_decorator
		def Template(params_):
			BasicEntanglerLayers(params_, wires=range(n_wires))

		template = Template

	return device, diff_method, interface, params, template, measurement
