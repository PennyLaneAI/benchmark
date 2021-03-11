# PennyLane Benchmarking

(Work in Progress)

This repository holds the benchmarking tools for [PennyLane](https://github.com/PennyLaneAI/pennylane).

Check out the current results at [https://pennylaneai.github.io/benchmark/](https://pennylaneai.github.io/benchmark/)

## Structure

`asv.conf.json`: configures asv.

`asv_command_list.rst`: relevant common commands.

`profiling_instructions.md`: an explanation of how to run profiling on a particular function.

`benchmarks/asv_benchmarks`: folder containing the benchmarks suites composed for ASV runs.

`benchmarks/benchmark_functions`: folder holding the basic benchmark functions which can be used independently of ASV.

`customenv_build.sh`: clones plugin source code into `.asv/sources`, creates a conda environment in `.asv/env/customenv`, populates the custom environment with necessary packages.

`update_sources.sh`: runs `git pull` on the plugins within `.asv/sources`

## Using benchmark functions without ASV

Here is an advanced example of how the benchmark functions can be customised and used in combination with 
Python's `time` library:

``` python

from benchmark_functions.circuit import benchmark_circuit
import time
import pennylane as qml
import numpy as np

mywires = ['a', 'b', 'c']
mydevice = qml.device('default.qubit.tf', wires=mywires)

@qml.templates.decorator.template
def MyTemplate(params_):
    qml.templates.ArbitraryUnitary(params_, wires=mywires)

myparams = np.random.random(4**3-1)
mymeasurement = qml.probs(['a', 'c'])

hp = {'n_wires': len(mywires), # this argument will not be used in this example
      'n_layers': 5, # this argument will not be used in this example
      'device': mydevice, # could also be a device name, in which case the wires are inferred from 'n_wires'
      'interface': 'tf',
      'diff_method': 'backprop',
      'template': MyTemplate,
      'params': myparams,
      'measurement': mymeasurement}


start = time.time()
benchmark_circuit(hyperparams=hp, num_repeats=10)
print(time.time()-start)

```

## Quickstart to run suites with ASV

The repository provides already configured collections of benchmark functions called "suites". These 
suites are automatically detected by ASV.

ASV can be installed via

`pip install asv `

To benchmark all suites for a single commit:

`asv run <commit>^..<commit>`

To run asv with the custom environment:

`asv run -E'existing:.asv/env/customenv/bin/python'`
  
To benchmark all suites for a commit range (see git revisions for more details):

`asv run <commit1>..<commit2>`
  
To benchmark selected commits stored in the provided commits.txt file:

`asv run HASHFILE:commits.txt`

Visualisation of commits is done by:

`asv publish`
`asv preview`

Single suites can be run by specifying a regular expression in the ``--bench`` argument.

More details can be found in the wonderful [asv docs](https://asv.readthedocs.io/en/stable/).

Contributors:
Christina Lee, Maria Schuld
