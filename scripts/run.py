"""
Runs a benchmark training of algorithms
    > python scripts/run.py --config <path/to/config>

"""
import argparse

from piasbenchmark import Benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a benchmark config")
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help="Path to the config file")
    args = parser.parse_args()

    benchmark = Benchmark(args.config)

    for job in benchmark.jobs():
        benchmark.run(job)
