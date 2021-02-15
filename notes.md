split and seed should be fixed for benchmark and not model dependant

Autokeras didn't work, didn't seem to output multilabels

Finished refactoring selectors and classifiers
Still need to do baselines

Then need to test to make sure it all works

Then need to run them all, models are now out of date unfortunatly

In baseline runner, make sure to use the splits properly

# Code for running it on horus
from tpotbench import Benchmark; b = Benchmark('./tpotbench/tpotbench/configs/benchmark1.0.json'); jobs = b.jobs(); statuts = b.status();

# Libraries
* ML-Plan is Java only (possible but tricky)
* PyAutoWeka outdated (2014), doesn't work for Python 3 (still possible)
* H20 doesn't support multilabel classification (not possible)
* Hyperoptsklearn doesn't support predict_proba (not possible)
* Autogluon (dependancy conflicts with Autosklearn, possible but tricky)
* Autokeras semi-works, as a selector, doesn't seem to like multi-label prediction
    as baseline, getting error 
    `tensorflow.python.framework.errors_impl.InvalidArgumentError: Value for attr 'T' of uint8 is not in the list of allowed values: int8, int16, int32, int64, complex64, complex128, float, double, bool`

    Under the impression this is due to the data set from the stacktrace

    `super().fit(...) -> self._analyze_data(dataset) -> error`

    It also breaks on the server
    ```
    import autokeras
    Illegal Instruction
    ```
    Stack overflow seems to say this is a common error to do with tensorflow.
    Solutions seem to be downgrade tensorflow (can't do with autokeras 
    depending on new versions) or compile it from source.

# Task errors
* Timeout 
    * T-TR_146825
    * bASK-14970
    * bASK-125920
    * ASK-18
    * ASK-219
    * ASK-2074
    * ASK-7592
    * ASK-9952
    * ASK-9960
    * ASK-9985
    * ASK-14952
    * ASK-14954
    * ASK-14969
    * ASK-14970
    * ASK-146195
    * ASK-146819
    * ASK-167119
    * ASK-167120
    * ASK-167125
* Memout:
   * ASK-12 (32gb > 24gb)
   * ASK-9910 (32gb > 24gb)
   * ASK-9964 (32gb > 24gb)
   * ASK-9981 (32gb > 24gb)
   * ASK-146824 (33gb > 24gb)

* Task 3021 seems to have some issues with
```
T-NB_3021_120_5
T-TR_3021_120_5
T-KNN_3021_120_5
T-MLP_3021_120_5
T-LR_3021_120_5
T-XGB_3021_120_5
bASK-3021_120_5

Getting the mode value in the util:
mode = X[col].mode()[0]
ValueError: 0 is not in range
```

* Task 167121
```
  File "/work/ws-tmp/eb130475-benchmark/.venv/lib/python3.8/site-packages/joblib/parallel.py", line 820, in dispatch_one_batch
    tasks = self._ready_batches.get(block=False)
  File "/home/eb130475/.pyenv/versions/3.8.6/lib/python3.8/queue.py", line 167, in get
    raise Empty
_queue.Empty

During handling of the above exception, another exception occurred:

...
ValueError: n_splits=5 cannot be greater than the number of members in each class.
```

* TPOT XGB on task 3573, 146825
```
UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].

 self._update_top_pipeline()

RuntimeError: There was an error in the TPOT optimization process. This could be because the data was not formatted properly, or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly. If you enabled PyTorch estimators, please check the data requirements in the online documentation: https://epistasislab.github.io/tpot/using/
```


