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
Define molecular Hamiltonians for VQE benchmarks.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane import Identity, PauliX, PauliY, PauliZ

############ H molecule #################################################

h2_coeffs = np.array(
    [
        -0.05963862,
        0.17575739,
        0.17575739,
        -0.23666489,
        -0.23666489,
        0.17001485,
        0.04491735,
        -0.04491735,
        -0.04491735,
        0.04491735,
        0.12222641,
        0.16714376,
        0.16714376,
        0.12222641,
        0.17570278,
    ]
)

h2_ops = [
    Identity(wires=[0]),
    PauliZ(wires=[0]),
    PauliZ(wires=[1]),
    PauliZ(wires=[2]),
    PauliZ(wires=[3]),
    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
]

ham_h2 = qml.Hamiltonian(h2_coeffs, h2_ops)

############# LiH #############################################################

lih_coeffs = np.array(
    [
        -6.74845266e00,
        -1.02553930e-01,
        1.00530907e-02,
        1.00530907e-02,
        -1.02553930e-01,
        1.00530907e-02,
        1.00530907e-02,
        -2.76355319e-01,
        -2.76355319e-01,
        -2.96925596e-01,
        -2.96925596e-01,
        -2.96925596e-01,
        -2.96925596e-01,
        1.21916192e-01,
        1.21233148e-02,
        1.21233148e-02,
        1.21233148e-02,
        1.21233148e-02,
        3.25324294e-03,
        -3.25324294e-03,
        -3.25324294e-03,
        3.25324294e-03,
        5.86266678e-03,
        -5.86266678e-03,
        -5.86266678e-03,
        5.86266678e-03,
        5.86266678e-03,
        -5.86266678e-03,
        -5.86266678e-03,
        5.86266678e-03,
        5.26857432e-02,
        5.59389862e-02,
        -1.85422006e-03,
        -1.85422006e-03,
        4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        4.81813200e-03,
        4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        4.81813200e-03,
        6.17431075e-02,
        3.39017831e-03,
        3.39017831e-03,
        6.76057742e-02,
        -1.42795369e-03,
        -1.42795369e-03,
        6.17431075e-02,
        3.39017831e-03,
        3.39017831e-03,
        6.76057742e-02,
        -1.42795369e-03,
        -1.42795369e-03,
        5.59389862e-02,
        -1.85422006e-03,
        -1.85422006e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        -4.81813200e-03,
        5.26857432e-02,
        6.76057742e-02,
        -1.42795369e-03,
        -1.42795369e-03,
        6.17431075e-02,
        3.39017831e-03,
        3.39017831e-03,
        6.76057742e-02,
        -1.42795369e-03,
        -1.42795369e-03,
        6.17431075e-02,
        3.39017831e-03,
        3.39017831e-03,
        8.44840116e-02,
        1.03194543e-02,
        -1.03194543e-02,
        -1.03194543e-02,
        1.03194543e-02,
        1.03194543e-02,
        -1.03194543e-02,
        -1.03194543e-02,
        1.03194543e-02,
        6.01815510e-02,
        7.05010052e-02,
        6.01815510e-02,
        7.05010052e-02,
        7.05010052e-02,
        6.01815510e-02,
        7.05010052e-02,
        6.01815510e-02,
        7.82363778e-02,
        4.21728488e-03,
        -4.21728488e-03,
        -4.21728488e-03,
        4.21728488e-03,
        6.55845232e-02,
        6.98018080e-02,
        6.98018080e-02,
        6.55845232e-02,
        7.82363778e-02,
    ]
)

lih_ops = [
    Identity(wires=[0]),
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
    PauliY(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliX(wires=[3])
    @ PauliX(wires=[4])
    @ PauliY(wires=[5]),
    PauliY(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliY(wires=[3])
    @ PauliX(wires=[4])
    @ PauliX(wires=[5]),
    PauliX(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliX(wires=[3])
    @ PauliY(wires=[4])
    @ PauliY(wires=[5]),
    PauliX(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliY(wires=[3])
    @ PauliY(wires=[4])
    @ PauliX(wires=[5]),
    PauliY(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliX(wires=[3])
    @ PauliX(wires=[6])
    @ PauliY(wires=[7]),
    PauliY(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliY(wires=[3])
    @ PauliX(wires=[6])
    @ PauliX(wires=[7]),
    PauliX(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliX(wires=[3])
    @ PauliY(wires=[6])
    @ PauliY(wires=[7]),
    PauliX(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliY(wires=[3])
    @ PauliY(wires=[6])
    @ PauliX(wires=[7]),
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
    PauliZ(wires=[6]) @ PauliZ(wires=[7]),
]

ham_lih = qml.Hamiltonian(lih_coeffs, lih_ops)
