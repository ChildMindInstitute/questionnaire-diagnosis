# Notebooks on data from the Child Mind Institute

##  Versioning with Jupyter

The following conventions hold:

1. Only notebooks with cleared output are committed.
2. Other notebooks should contain the string `WIP` and are thus ignored by `git`.
3. Notebooks that mark a meeting or other kind of checkpoint should contain a string with date that must include a year. Those are also ignored by `git`.
4. Older notebooks (for completeness) can be dumped into the directory `old-notebooks` which is also ignored by `git`.
5. `git` also ignores the resources directory because it contains proprietary questionnaire information and private data sets.


##  Questionnaire shortening

Initialize the submodule:
```
$ git init submodule
```
change directory to the submodule:
```
$ cd to quantify-predictive-value
```
Run a smoke test on the childmind data:
```
$ pytest src/quantify_predictive_value.py --path-to-config ../config-files/smoke-test-config.json
```.
Run an actual experiment:
```
$ pytest src/quantify_predictive_value.py --path-to-config ../config-files/childmind-config.json --replicates 50 -n 50
```.
