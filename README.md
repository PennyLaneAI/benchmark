# PennyLane Benchmarking

(Work in Progress)

This repository holds the benchmarking tools for [PennyLane](https://github.com/PennyLaneAI/pennylane).

Check out the current results at [https://pennylaneai.github.io/benchmark/](https://pennylaneai.github.io/benchmark/)

## Structure

`asv.conf.json`: configures asv.

`asv_command_list.rst`: relevant common commands.

`profiling_instructions.md`: an explanation of how to run profiling on a particular function.

`benchmarks/asv_benchmarks`: folder holding the benchmark cases

`benchmarks/benchmark_functions`: folder holding the functions that cases execute.  Functions independently to asv

## Quickstart

To run the benchmarks, install avs via

pip install asv 

To benchmark a single commit:

asv run <commit>^..<commit>
  
To benchmark all commits that connect two commits:

asv run <commit1>..<commit2>
  
To benchmark selected commits stored in the provided commits.txt file:

asv run HASHFILE:commits.txt

Visualisation of commits is done by:

asv publish
asv preview

More details can be found in the wonderful [asv docs](https://asv.readthedocs.io/en/stable/).

Contributors:
Christina Lee, Maria Schuld
