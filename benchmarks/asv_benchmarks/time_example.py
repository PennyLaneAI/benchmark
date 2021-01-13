from ..benchmark_functions.benchmark_example_file import BenchmarksExample_Class

class Suite_example:


    "These are the parameters we run the test for"
    params = [1e-6, 1e-4, 1e-2]

    def setup(self, duration):
        """ 
        A setup function allows us to perform initial comptutations like 
        creating data structures before the timing begins.

        We can create or destroy class variables for use in the time function.
        """
        pass

    def teardown(self, duration):
        """
        Like the setup function, teardown allows us to perform non-timed
        processing steps, like writing outputs or garbage collection
        """
        pass

    def time_example(self, duration):
        """
        The preface `time` indicates we are tracking time
        """
        result = BenchmarksExample_Class.benchmark_example_function(duration)

    def time_example_here(self, class_param):
        """
        We can have multiple benchmarks in the same suite.
        They will use the same parameters and same setup and teardown functions.

        This is a bad function to place here, because it doesn't depend on the parameters,
        yet still gets run multiple times.
        """
        print("hello world")

class Suite_example_mem:
    
    # Now we have two parameters to perform a grid evaluation over
    params = ([0, 11, 12345], [1000, 10000])
    
    # we can also give names to the different parameters
    param_names = ['val', 'n']

    # We don't need to create setup and teardown functions for
    # every case.  So let's skip those here

    def mem_example(self, val, n):
        """
        The preface `mem` calculates the size of the returned value
        """
        return BenchmarksExample_Class.benchmark_example_mem_function(val, n)

    def peakmem_example(self, val, n):
        """
        This preface `mempeak` indicates we are tracking peak memory.
        It calculates the maximum resident size in bytes of the process
        """
        results = BenchmarksExample_Class.benchmark_example_mem_function(val, n)
