# benchmark
Benchmarks the PennyLane library using airspeed velocity. At the moment the benchmarks are just dummies. Add new ones by adding methods to the files in the `benchmarks` folder.

To run the benchmarks, install avs via

```console
pip install asv 
```

To benchmark a single commit:

```console
asv run <commit>^..<commit>
```

To benchmark all commits that connect two commits:

```console
asv run <commit1>..<commit2>
```

To benchmark selected commits stored in the provided `commits.txt` file:

```console
asv run HASHFILE:commits.txt
```

Visualisation of commits is done by

```console
asv publish
asv preview
```



More details can be found in the wonderful [asv docs](https://asv.readthedocs.io/en/stable/).
