mkdir .asv/sources
git clone https://github.com/PennyLaneAI/pennylane-lightning.git .asv/sources/pennylane-lightning
git clone https://github.com/PennyLaneAI/pennylane-qiskit.git .asv/sources/pennylane-qiskit
git clone https://github.com/PennyLaneAI/pennylane-cirq.git .asv/sources/pennylane-cirq
git clone https://github.com/PennyLaneAI/pennylane-qulacs.git .asv/sources/pennylane-qulacs

conda create -y --prefix ./.asv/env/customenv python=3.8
yes | .asv/env/customenv/bin/python -m pip install -e .asv/sources/pennylane-lightning
yes | .asv/env/customenv/bin/python -m pip install -e .asv/sources/pennylane-qiskit
yes | .asv/env/customenv/bin/python -m pip install -e .asv/sources/pennylane-cirq
yes | .asv/env/customenv/bin/python -m pip install -e .asv/sources/pennylane-qulacs

yes | .asv/env/customenv/bin/python -m pip install tensorflow
yes | .asv/env/customenv/bin/python -m pip install torch
yes | .asv/env/customenv/bin/python -m pip install networkx
yes | .asv/env/customenv/bin/python -m pip install Qulacs
yes | .asv/env/customenv/bin/python -m pip install qsimcirq


git clone https://github.com/PennyLaneAI/pennylane.git .asv/env/customenv/project
yes | .asv/env/customenv/bin/python -m pip install -e .asv/env/customenv/project/qchem
