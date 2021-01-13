from ..benchmark_functions.benchmark_example import BenchmarksExample

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
        This function is prefaced with `time` to indicate we are tracking
        time
        """
        result = BenchmarksExample.benchmark_example(duration)

class Suite_example_mem:
    
    # Now we have two parameters to perform a grid evaluation over
    params = ([0, 11, 12345], [1000, 10000])
    
    # we can also give names to the different parameters
    param_names = ['val', 'n']

    # We don't need to create setup and teardown functions for
    # every case.  So let's skip those here

    def mem_example(self, val, n):
        """
        this function is prefaced `mem` to indicate we are tracking memory
        """
        results = BenchmarksExample.benchmark_example_mem(val, n)
