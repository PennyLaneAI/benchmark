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
Define asv benchmark suite that estimates the speed of applications.
"""

from ..benchmark_functions.vqe import benchmark_vqe


class VQEEvaluation:
    """Benchmark VQE."""

    params = ([1, 3], [False, True], ['best', 'parameter-shift'])
    param_names = ['n_steps', 'optimize', 'diff_method']

    def time_hydrogen(self, n_steps, optimize, diff_method):
        """Time VQE for the hydrogen molecule."""
        hyperparams = {'n_steps': n_steps,
                       'optimize': optimize,
                       'diff_method': diff_method}
        benchmark_vqe(hyperparams)
