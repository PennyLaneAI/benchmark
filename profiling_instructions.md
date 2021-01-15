# Profiling

Profiling can be a bit tricky to get working.

Instead of a general format, here is an example:

`asv profile 'asv_benchmarks.time_example.Suite_example.time_example\(0.01\)' a2fa39c --output profiles/profile_file.prof`

The first variable is the full function name.  Include `\` around parentheses.  The parameter provided *must* be one that exists in the Suite and has been benchmarked already. For example, `10` would give errors as that value is not in `Suite_example.params`.

The next variable, `a2fa39c` is the hash of the commit to run the profile on.  Make sure you have already benchmarked this commit.

The `--output` flag designates the file where the profiling data gets saved to.

Once you have the data saved, you can view it in a program like [snakeviz](https://jiffyclub.github.io/snakeviz/).