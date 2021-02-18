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
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines import UCCSD
from functools import partial
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import qchem


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

    def peakmem_hydrogen(self, n_steps, optimize):
        """Time a VQE algorithm with the UCCSD ansatz for computing the ground state energy of the
         hydrogen molecule."""
        hyperparams = {'n_steps': n_steps,
                       'optimize': optimize}
        benchmark_vqe(hyperparams)


class heavyVQE:
    """Benchmark the VQE algorithm using different number of optimization steps and grouping
     options."""

    params = ([3], [False, True])
    param_names = ['n_steps', 'optimize']

    H_coeffs = np.array([-2.73959876e-01,  2.18445899e-01, -2.48874139e-07, -2.48874139e-07,
    2.46851443e-07,  2.46851443e-07,  2.18445899e-01, -2.48874139e-07,
    -2.48874139e-07,  2.46851443e-07,  2.46851443e-07, -1.19231631e-01,
    -1.19231631e-01, -1.19242832e-01, -1.19242832e-01,  1.48099521e-01,
    6.13904082e-07,  6.13904082e-07, -6.08858545e-07, -6.08858545e-07,
    6.13904082e-07,  6.13904082e-07,  3.59786739e-02, -3.59786739e-02,
    -3.59786739e-02,  3.59786739e-02, -6.08858545e-07, -6.08858545e-07,
    3.59786994e-02, -3.59786994e-02, -3.59786994e-02,  3.59786994e-02,
    1.07944533e-01, -5.40112486e-07, -5.40112486e-07,  1.43923207e-01,
    -1.67338505e-02, -1.67338505e-02,  1.65959398e-02,  1.65959398e-02,
    -1.65964799e-02, -1.65964799e-02, -1.65964799e-02, -1.65964799e-02,
    -1.67336289e-02,  1.67336289e-02,  1.67336289e-02, -1.67336289e-02,
    1.07945494e-01,  5.44579783e-07,  5.44579783e-07,  1.43924193e-01,
    1.67341735e-02,  1.67341735e-02, -1.65962602e-02, -1.65962602e-02,
    1.43923207e-01, -1.67338505e-02, -1.67338505e-02,  1.65959398e-02,
    1.65959398e-02, -1.65964799e-02,  1.65964799e-02,  1.65964799e-02,
    -1.65964799e-02,  1.67336289e-02,  1.67336289e-02,  1.67336289e-02,
    1.67336289e-02,  1.07944533e-01, -5.40112486e-07, -5.40112486e-07,
    1.43924193e-01,  1.67341735e-02,  1.67341735e-02, -1.65962602e-02,
    -1.65962602e-02,  1.07945494e-01,  5.44579783e-07,  5.44579783e-07,
    1.64602317e-01,  4.58772679e-07,  4.58772679e-07,  4.58772679e-07,
    4.58772679e-07,  1.87299262e-02, -1.87299262e-02, -1.87299262e-02,
    1.87299262e-02,  1.08412748e-01,  1.27142674e-01, -4.58769212e-07,
    -4.58769212e-07,  1.27142674e-01, -4.58769212e-07, -4.58769212e-07,
    1.08412748e-01,  1.64602766e-01])

    H_ops = [Identity(wires=[0]),
         PauliZ(wires=[0]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(
             wires=[4]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(
             wires=[4]),
         PauliZ(wires=[1]),
         PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]),
         PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]),
         PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(
             wires=[5]),
         PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(
             wires=[5]),
         PauliZ(wires=[2]),
         PauliZ(wires=[3]),
         PauliZ(wires=[4]),
         PauliZ(wires=[5]),
         PauliZ(wires=[0]) @ PauliZ(wires=[1]),
         PauliY(wires=[0]) @ PauliY(wires=[2]),
         PauliX(wires=[0]) @ PauliX(wires=[2]),
         PauliY(wires=[0]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]),
         PauliX(wires=[0]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]),
         PauliZ(wires=[0]) @ PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]),
         PauliZ(wires=[0]) @ PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]),
         PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
         PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
         PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
         PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
         PauliZ(wires=[0]) @ PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(
             wires=[4]) @ PauliY(wires=[5]),
         PauliZ(wires=[0]) @ PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(
             wires=[4]) @ PauliX(wires=[5]),
         PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
         PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
         PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
         PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
         PauliZ(wires=[0]) @ PauliZ(wires=[2]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]),
         PauliZ(wires=[0]) @ PauliZ(wires=[3]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[3]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[3]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[4]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[4]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(
             wires=[4]) @ PauliY(wires=[5]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(
             wires=[4]) @ PauliX(wires=[5]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(
             wires=[4]) @ PauliY(wires=[5]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(
             wires=[4]) @ PauliX(wires=[5]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliX(
             wires=[4]) @ PauliY(wires=[5]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliX(
             wires=[4]) @ PauliX(wires=[5]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliY(
             wires=[4]) @ PauliY(wires=[5]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliY(
             wires=[4]) @ PauliX(wires=[5]),
         PauliZ(wires=[0]) @ PauliZ(wires=[4]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[4]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[4]),
         PauliZ(wires=[0]) @ PauliZ(wires=[5]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[5]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[5]),
         PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(
             wires=[4]) @ PauliZ(wires=[5]),
         PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(
             wires=[4]) @ PauliZ(wires=[5]),
         PauliZ(wires=[1]) @ PauliZ(wires=[2]),
         PauliY(wires=[1]) @ PauliY(wires=[3]),
         PauliX(wires=[1]) @ PauliX(wires=[3]),
         PauliY(wires=[1]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
         PauliX(wires=[1]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
         PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[4]),
         PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[4]),
         PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[4]),
         PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[4]),
         PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
         PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
         PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
         PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
         PauliZ(wires=[1]) @ PauliZ(wires=[3]),
         PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
         PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
         PauliZ(wires=[1]) @ PauliZ(wires=[4]),
         PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[4]),
         PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[4]),
         PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[5]),
         PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[5]),
         PauliZ(wires=[1]) @ PauliZ(wires=[5]),
         PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[5]),
         PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[5]),
         PauliZ(wires=[2]) @ PauliZ(wires=[3]),
         PauliY(wires=[2]) @ PauliY(wires=[4]),
         PauliX(wires=[2]) @ PauliX(wires=[4]),
         PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
         PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
         PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
         PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
         PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
         PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
         PauliZ(wires=[2]) @ PauliZ(wires=[4]),
         PauliZ(wires=[2]) @ PauliZ(wires=[5]),
         PauliY(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]) @ PauliZ(wires=[5]),
         PauliX(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]) @ PauliZ(wires=[5]),
         PauliZ(wires=[3]) @ PauliZ(wires=[4]),
         PauliY(wires=[3]) @ PauliY(wires=[5]),
         PauliX(wires=[3]) @ PauliX(wires=[5]),
         PauliZ(wires=[3]) @ PauliZ(wires=[5]),
         PauliZ(wires=[4]) @ PauliZ(wires=[5])]

    Hamiltonian = qml.Hamiltonian(H_coeffs, H_ops)

    electrons = 2
    qubits = 6

    singles, doubles = qchem.excitations(electrons, qubits)
    s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
    hf_state = qchem.hf_state(electrons, qubits)

    ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)
    parameters = np.array(
        [-0.78814194, 2.06032181, -4.57916617, 1.37300652, -5.14208346, 0.01745228, 3.41288487,
         -2.01876549])

    device = qml.device('default.qubit', wires=qubits)


    def peakmem_trihydrogen(self, n_steps, optimize):
        """Time"""

        # H_coeffs = np.array([-2.73959876e-01,  2.18445899e-01, -2.48874139e-07, -2.48874139e-07,
        # 2.46851443e-07,  2.46851443e-07,  2.18445899e-01, -2.48874139e-07,
        # -2.48874139e-07,  2.46851443e-07,  2.46851443e-07, -1.19231631e-01,
        # -1.19231631e-01, -1.19242832e-01, -1.19242832e-01,  1.48099521e-01,
        # 6.13904082e-07,  6.13904082e-07, -6.08858545e-07, -6.08858545e-07,
        # 6.13904082e-07,  6.13904082e-07,  3.59786739e-02, -3.59786739e-02,
        # -3.59786739e-02,  3.59786739e-02, -6.08858545e-07, -6.08858545e-07,
        # 3.59786994e-02, -3.59786994e-02, -3.59786994e-02,  3.59786994e-02,
        # 1.07944533e-01, -5.40112486e-07, -5.40112486e-07,  1.43923207e-01,
        # -1.67338505e-02, -1.67338505e-02,  1.65959398e-02,  1.65959398e-02,
        # -1.65964799e-02, -1.65964799e-02, -1.65964799e-02, -1.65964799e-02,
        # -1.67336289e-02,  1.67336289e-02,  1.67336289e-02, -1.67336289e-02,
        # 1.07945494e-01,  5.44579783e-07,  5.44579783e-07,  1.43924193e-01,
        # 1.67341735e-02,  1.67341735e-02, -1.65962602e-02, -1.65962602e-02,
        # 1.43923207e-01, -1.67338505e-02, -1.67338505e-02,  1.65959398e-02,
        # 1.65959398e-02, -1.65964799e-02,  1.65964799e-02,  1.65964799e-02,
        # -1.65964799e-02,  1.67336289e-02,  1.67336289e-02,  1.67336289e-02,
        # 1.67336289e-02,  1.07944533e-01, -5.40112486e-07, -5.40112486e-07,
        # 1.43924193e-01,  1.67341735e-02,  1.67341735e-02, -1.65962602e-02,
        # -1.65962602e-02,  1.07945494e-01,  5.44579783e-07,  5.44579783e-07,
        # 1.64602317e-01,  4.58772679e-07,  4.58772679e-07,  4.58772679e-07,
        # 4.58772679e-07,  1.87299262e-02, -1.87299262e-02, -1.87299262e-02,
        # 1.87299262e-02,  1.08412748e-01,  1.27142674e-01, -4.58769212e-07,
        # -4.58769212e-07,  1.27142674e-01, -4.58769212e-07, -4.58769212e-07,
        # 1.08412748e-01,  1.64602766e-01])
        # H_coeffs = self.H_coeffs
        # H_ops = [Identity(wires=[0]),
        #  PauliZ(wires=[0]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]),
        #  PauliZ(wires=[1]),
        #  PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]),
        #  PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]),
        #  PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[2]),
        #  PauliZ(wires=[3]),
        #  PauliZ(wires=[4]),
        #  PauliZ(wires=[5]),
        #  PauliZ(wires=[0]) @ PauliZ(wires=[1]),
        #  PauliY(wires=[0]) @ PauliY(wires=[2]),
        #  PauliX(wires=[0]) @ PauliX(wires=[2]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]),
        #  PauliZ(wires=[0]) @ PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]),
        #  PauliZ(wires=[0]) @ PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]),
        #  PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
        #  PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
        #  PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
        #  PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
        #  PauliZ(wires=[0]) @ PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
        #  PauliZ(wires=[0]) @ PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
        #  PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
        #  PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
        #  PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[0]) @ PauliZ(wires=[2]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]),
        #  PauliZ(wires=[0]) @ PauliZ(wires=[3]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[3]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[3]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[4]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[4]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[0]) @ PauliZ(wires=[4]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[4]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[4]),
        #  PauliZ(wires=[0]) @ PauliZ(wires=[5]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliY(wires=[2]) @ PauliZ(wires=[5]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliX(wires=[2]) @ PauliZ(wires=[5]),
        #  PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]) @ PauliZ(wires=[5]),
        #  PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]) @ PauliZ(wires=[5]),
        #  PauliZ(wires=[1]) @ PauliZ(wires=[2]),
        #  PauliY(wires=[1]) @ PauliY(wires=[3]),
        #  PauliX(wires=[1]) @ PauliX(wires=[3]),
        #  PauliY(wires=[1]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[1]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
        #  PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[4]),
        #  PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[4]),
        #  PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[4]),
        #  PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[4]),
        #  PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
        #  PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
        #  PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[1]) @ PauliZ(wires=[3]),
        #  PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[1]) @ PauliZ(wires=[4]),
        #  PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[4]),
        #  PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[4]),
        #  PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[5]),
        #  PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[1]) @ PauliZ(wires=[5]),
        #  PauliY(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[5]),
        #  PauliX(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[5]),
        #  PauliZ(wires=[2]) @ PauliZ(wires=[3]),
        #  PauliY(wires=[2]) @ PauliY(wires=[4]),
        #  PauliX(wires=[2]) @ PauliX(wires=[4]),
        #  PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliZ(wires=[4]) @ PauliY(wires=[5]),
        #  PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliZ(wires=[4]) @ PauliX(wires=[5]),
        #  PauliY(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
        #  PauliY(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
        #  PauliX(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
        #  PauliX(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[2]) @ PauliZ(wires=[4]),
        #  PauliZ(wires=[2]) @ PauliZ(wires=[5]),
        #  PauliY(wires=[2]) @ PauliZ(wires=[3]) @ PauliY(wires=[4]) @ PauliZ(wires=[5]),
        #  PauliX(wires=[2]) @ PauliZ(wires=[3]) @ PauliX(wires=[4]) @ PauliZ(wires=[5]),
        #  PauliZ(wires=[3]) @ PauliZ(wires=[4]),
        #  PauliY(wires=[3]) @ PauliY(wires=[5]),
        #  PauliX(wires=[3]) @ PauliX(wires=[5]),
        #  PauliZ(wires=[3]) @ PauliZ(wires=[5]),
        #  PauliZ(wires=[4]) @ PauliZ(wires=[5])]
        #
        # Hamiltonian = qml.Hamiltonian(H_coeffs, H_ops)
        #
        # electrons = 2
        # qubits = 6
        #
        # singles, doubles = qchem.excitations(electrons, qubits)
        # s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
        # hf_state = qchem.hf_state(electrons, qubits)
        #
        # ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)
        # parameters = np.array([-0.78814194, 2.06032181, -4.57916617, 1.37300652, -5.14208346, 0.01745228, 3.41288487, -2.01876549])
        #
        # device = qml.device('default.qubit', wires=qubits)

        hyperparams = {'Hamiltonian': self.Hamiltonian,
                       'ansatz': self.ansatz,
                       'params': self.parameters,
                       'device': self.device,
                       'n_steps': n_steps,
                       'optimize': optimize}

        benchmark_vqe(hyperparams)


    def peakmem_lih(self, n_steps, optimize):
        """Time"""

        H_coeffs = np.array([-6.74845266e+00, -1.02553930e-01,  1.00530907e-02,  1.00530907e-02,
       -1.02553930e-01,  1.00530907e-02,  1.00530907e-02, -2.76355319e-01,
       -2.76355319e-01, -2.96925596e-01, -2.96925596e-01, -2.96925596e-01,
       -2.96925596e-01,  1.21916192e-01,  1.21233148e-02,  1.21233148e-02,
        1.21233148e-02,  1.21233148e-02,  3.25324294e-03, -3.25324294e-03,
       -3.25324294e-03,  3.25324294e-03,  5.86266678e-03, -5.86266678e-03,
       -5.86266678e-03,  5.86266678e-03,  5.86266678e-03, -5.86266678e-03,
       -5.86266678e-03,  5.86266678e-03,  5.26857432e-02,  5.59389862e-02,
       -1.85422006e-03, -1.85422006e-03,  4.81813200e-03, -4.81813200e-03,
       -4.81813200e-03,  4.81813200e-03,  4.81813200e-03, -4.81813200e-03,
       -4.81813200e-03,  4.81813200e-03,  6.17431075e-02,  3.39017831e-03,
        3.39017831e-03,  6.76057742e-02, -1.42795369e-03, -1.42795369e-03,
        6.17431075e-02,  3.39017831e-03,  3.39017831e-03,  6.76057742e-02,
       -1.42795369e-03, -1.42795369e-03,  5.59389862e-02, -1.85422006e-03,
       -1.85422006e-03, -4.81813200e-03, -4.81813200e-03, -4.81813200e-03,
       -4.81813200e-03, -4.81813200e-03, -4.81813200e-03, -4.81813200e-03,
       -4.81813200e-03,  5.26857432e-02,  6.76057742e-02, -1.42795369e-03,
       -1.42795369e-03,  6.17431075e-02,  3.39017831e-03,  3.39017831e-03,
        6.76057742e-02, -1.42795369e-03, -1.42795369e-03,  6.17431075e-02,
        3.39017831e-03,  3.39017831e-03,  8.44840116e-02,  1.03194543e-02,
       -1.03194543e-02, -1.03194543e-02,  1.03194543e-02,  1.03194543e-02,
       -1.03194543e-02, -1.03194543e-02,  1.03194543e-02,  6.01815510e-02,
        7.05010052e-02,  6.01815510e-02,  7.05010052e-02,  7.05010052e-02,
        6.01815510e-02,  7.05010052e-02,  6.01815510e-02,  7.82363778e-02,
        4.21728488e-03, -4.21728488e-03, -4.21728488e-03,  4.21728488e-03,
        6.55845232e-02,  6.98018080e-02,  6.98018080e-02,  6.55845232e-02,
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
        PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[4]) @ PauliY(wires=[5]),
        PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[4]) @ PauliX(wires=[5]),
        PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[4]) @ PauliY(wires=[5]),
        PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[4]) @ PauliX(wires=[5]),
        PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliX(wires=[6]) @ PauliY(wires=[7]),
        PauliY(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliX(wires=[6]) @ PauliX(wires=[7]),
        PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliX(wires=[3]) @ PauliY(wires=[6]) @ PauliY(wires=[7]),
        PauliX(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliY(wires=[3]) @ PauliY(wires=[6]) @ PauliX(wires=[7]),
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

        Hamiltonian = qml.Hamiltonian(H_coeffs, H_ops)

        electrons = 2
        qubits = 8

        singles, doubles = qchem.excitations(electrons, qubits)
        s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
        hf_state = qchem.hf_state(electrons, qubits)

        ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

        parameters = np.array([6.39225682, -0.99471664, -4.2026237 , -4.48579097,  9.8033157 ,
         1.19030864, -3.89924719,  7.25037131, -0.95897967, -0.75287453,
         0.92252162,  1.10633277,  0.94911997,  1.09138887,  5.27297259])

        device = qml.device('default.qubit', wires=qubits)

        hyperparams = {'Hamiltonian': Hamiltonian,
                       'ansatz': ansatz,
                       'params': parameters,
                       'device': device,
                       'n_steps': n_steps,
                       'optimize': optimize}

        benchmark_vqe(hyperparams)
