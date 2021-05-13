# Per Instance Algorithm Selection Benchmark
This benchmark currently uses two AutoML tools, [autosklearn](https://automl.github.io/auto-sklearn/master/) and [TPOT](http://epistasislab.github.io/tpot/)
and compares them with the state of the art algorithm _Multi-Classifier Systems_ available from [DESLib](https://github.com/scikit-learn-contrib/DESlib).

Trained models are not provided due to size, however their results are provided
and a smaller sample run of the benchmark is also easily accessible.

The benchmarking is split into three distinct phases
* Creating a benchmark config
* Training models
* Generating results from these models

This benchmark was run using a slurm cluster so support for slurm and local
running is available.

For any issues we are aware of, please see [Issues](#Issues)

## TLDR;
```sh
# ...in the cloned and installed git repo
python ./scripts/run.py --config ./configs/local_test.json    # ~30 min

# Create a place to store results
mkdir my_results_folder

# Generate the results to the folder
python ./scripts/gen_results.py \
    --config ./configs/local_test.json \
    --results ./my_results_folder

# Create the plots for the benchmark
python ./scripts/create_plots.py \
    --config ./configs/local_test.json \
    --results ./my_results_folder
    
```

## Installation
For installing, first the repository must be downloaded and then the dependancies listed in `requirements.txt`.
We recommend setting up a [virtual environment](https://docs.python.org/3/library/venv.html)  before installing to prevent conflicts from any other libraries
on your system.

```sh
# Get source code
git clone https://github.com/jonathandoe-submission/PerInstanceAlgorithmSelectionBenchmark/

# Enter repository
cd PerInstanceAlgorithmSelectionBenchmark

# Install all dependancies as listed in 
pip install -r requirements.txt

pip install -e .  # -e flag is so source code can be edited without reinstalling
```

This was tested with `Python 3.8.6`, for full compatibility, we recommend [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/en/latest/)
managing different Python versions.


## Configs
Two configs can be found in the `configs` folder.
* `configs/benchmark1.0.json`
* `configs/local_test.json`

`benchmark1.0.json` is unfeasible to run on a single machine and was run over
a slurm compute cluster, however see `local_test.json` as a quick 30 min example.

These are generated with the associated `configs/benchmark1.0.py` and
`configs/local_bench.py`, which are the best way to view how the ocnfigs are
generated

The configs allow you to specify the details:
* `seed` - The seed to use throughout.
* `split`- A three-tuple (algotrain, metatrain, test) to determine split sizes.
* `tasks`- A list of [OpenML](https://www.openml.org/) tasks to run on.
* `path` - Where all generated models and results are stored.
* `classifier` - A list of classifier configurations to run.
* `selector` - A list of selector configurations to run.
* `baseline` - A list of baseline algorithms to run.

Please see the provided configurations for more information.

## Running the benchmark
A benchmark config can be run using `scripts/run.py`

```sh
python ./scripts/run.py --config ./configs/local_test.json    # ~30 min
```

The models will be stored in the `path` option set by the config. These models
and folders are used to track state. Essentially this means that if a folder
`{path}/{job_name}` exists, the job has alreay been run and if a `model.pkl` 
exists within this folder, it was successful.

Logs for each job can also be found in these folders.

## Generating Results and Plots
Results can be generated with

```sh
# Create a directory to store results in
mkdir my_results_folder

# Generate the results to the folder
python ./scripts/gen_results.py \
    --config ./configs/local_test.json \
    --results ./my_results_folder
```

The following results are created and also provided for the primary experiment
* `{benchname}_results.json` - Per-task results for all models as `.json`
* `{benchname}_summary.json` - Averaged results for all models as `.json`
* `{benchname}_accuracies.csv` - Accuracies for all models on all datasets.
* `{benchname}_normscores.csv` - Normalized scores for all models on all datasets.

Furthermore, the plots can be generated with:
```sh
# Create the plots for the benchmark
python ./scripts/create_plots.py \
    --config ./configs/local_test.json \
    --results ./my_results_folder
```

This creates 6 plots (3 jpgs, 3 svgs) and 1 csv:
* `{benchname}_cached_metaprops.csv` - OpenML meta-features for datasets, cached for reproducibility.
* `{benchname}_centered.[jpg/svg]` - Plot for comparison between selectors and baselines.
* `{benchname}_umap_projections_[all/selectors_and_baselines].[jpg/svg]`
    * UMAP projection based on meta-feature distance, coloured by the best
        algorithm where `all` is all algorithms and `selectors_and_baselines`
        only colors by best selector or baseline

## Local versus Slurm
This work was done on a HPC that is running slurm and as such,
supports running model training either locally or distributed throughout the cluster.

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

As every slurm configuration will be different, please see `piasbenchmark/slurm.py`
for slurm parameters for your own slurm HPC.

# Issues
#### Please call fit() first
---
```
RuntimeError: A pipeline has not yet been optimized. Please call fit() first.

subprocess.CalledProcessError: Command '['python', '/home/user/piasbenchmark/runners/classifier_runner.py', '/home/user/piasbenchmark/local_test/T-TR_6_1_5/config.json']' returned non-zero exit status 1.
```

Usually occurs if a training of a classifier was interupted, the simplest
fix is to delete the associated folder,
in this case `~/piasbenchmark/local_test/T-TR-6-1-5`, and re-run it.

The benchmark checks the existence of these folders to see if a model has been
trained.

Known reasons for interuption are timeouts and memory allocation exceptions.
* Timeouts occur most often when server time allocations are too low to complete
model training.
    * A good rule of thumb is allocate 1.5x time required by `autosklearn`
or `tpot`. Please see `piasbenchmark/slurm` which automatically provides this buffer.
For other distributed computing software, please adapt as required.
* Memory issues can occur with larger datasets. The simplest solution is to 
    provide more memory for those specific datasets or to exclude them if 
    that is not possible.

#### smac attempted to use a functionality that requires module ...
---
```
ImportError: smac attempted to use a functionality that requires module pyDOE, but it couldn't be loaded. ...
```
Pip didn't install [smac's](https://github.com/automl/SMAC3) extra requires.

Manually install them with `pip install emcee pyDOE scikit-optimize`
