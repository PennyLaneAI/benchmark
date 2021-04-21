import time


class BenchmarksExample_Class:
    @staticmethod
    def benchmark_example_function(duration=0.001):
        time.sleep(duration)
        return True

    @staticmethod
    def benchmark_example_mem_function(val, n):
        return [val] * n
