**Note to Reviewer**: This is kept anonymized as best as possible by double blind submission regulations of the ICLR 2021. 
There will be some small updates in the official repository, notably code documentation, documentation on extension and a command line interface for easier running.

# Per Instance Algorithm Selection Benchmark
This benchmark currently uses two AutoML tools, [autosklearn](https://automl.github.io/auto-sklearn/master/) and [TPOT](http://epistasislab.github.io/tpot/)
and compares them with the state of the art algorithm _Multi-Classifier Systems_ available from [DESLib](https://github.com/scikit-learn-contrib/DESlib).

## Installation
For installing, first the repository must be downloaded and then the dependancies listed in `requirements.txt`. 
We recommend setting up a [virtual environment](https://docs.python.org/3/library/venv.html)  before installing to prevent conflicts from any other libraries
on your system.

```BASH
git clone https://github.com/jonathandoe-submission/PerInstanceAlgorithmSelectionBenchmark/
cd PerInstanceAlgorithmSelectionBenchmark
pip install -r requirements.txt
```

This was tested with `Python 3.8.6`, for full compatibility, we recommend [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/en/latest/)
managing different Python versions.

## Configs
The full benchmark configuration can be found at `src/configs/benchmark1.0.json` but as this is impractical to run on a local machine, we have also provided
a config for running on a single dataset `src/configs/local_bench.json`.
These are generated with the associated `src/configs/benchmark1.0.py` and `src/configs/local_bench.py`.

The configs allow you to specify the details:
* `seed` - The seed to use throughout.
* `split`- A three-tuple (algotrain, metatrain, test) to determine split sizes.
* `tasks`- A list of [OpenML](https://www.openml.org/) tasks to run on.
* `path` - Where all generated models and results are stored.
* `classifier` - A list of classifier configurations to run.
* `selector` - A list of selector configurations to run.
* `baseline` - A list of baseline algorithms to run.

Please see the provided configurations for examples.

## Running the benchmark
The benchmarking is split into two distinct phases
* Training and saving models
* Generating results
We recommend using the code interactively with [Ipython](https://ipython.org/) or [Jupyter notebooks](https://jupyter.org/).


To train and save the various models
```Python
from PerInstanceAlgorithmSelectionBenchmark import Benchmark

bench = Benchmark('path/to/src/configs/local_test.json')
bench.run()

# Alternatively, this can be done more explicitly
jobs = bench.jobs()
... filter the jobs
bench.run(jobs)
```

# Generating results
To generate the results, simply run
```
python src/gen_results.py <path/to/benchmark/config>
```

The results will be stored as `results.json` in the `path` specified by the config.

## Local versus Slurm
This work was done on a HPC that is running Slurm and as such, supports running model training either locally or distributed throughout the cluster.

This is specified in the configuration files by:
```Python
benchmark_config = {
  ...,
  # For slurm
  env : { 
    'type' : 'slurm',
    'username': 'jonathan doe' # Your username on the slurm cluster 
  },

  # For local
  env : {
    'type': 'local'
  }
```

As every slurm configuration will be different, please see `src/slurm.py` for slurm parameters for your own slurm HPC.
