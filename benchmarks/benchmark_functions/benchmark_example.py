import time

class BenchmarksExample:

    @staticmethod
    def benchmark_example(duration=0.001):
        time.sleep(duration)
        return True

    @staticmethod
    def benchmark_example_mem(val, n):
        return [val] * n