=================
asv commands list
=================

Ways to Run
-----------

`asv run`: run latest commit on current master branch

`asv run --quick`: benchmarks are run only once.  good for debugging

`asv run NEW`: benchmark all commits since the last one on this machine

`asv run HASHFILE:hashestobenchmark.txt`: run commits explicitly listed


`asv dev`: run once, show errors, and don't save

`asv run --bench words`: runs only benchmarks with `words` in their name. For example, `Suite_example` runs all tests in `Suite_example` and `Suite_example_mem`.  For another case, `example_here` runs only `Suite_example.time_example_here`.



Viewing Results
---------------

`asv show`: print saved benchmark results to command line

`asv publish`: create the html directory

`asv preview`: locally serve the html directory

`asv gh-pages`: update gh-pages branch

Profiling
---------

Have yet to figure out how this works