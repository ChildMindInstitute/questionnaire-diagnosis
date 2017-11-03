#!/bin/sh

set -Ceu

: ${PYTHON:=python}

root=`cd -- "$(dirname -- "$0")" && pwd`

# XXX I bet there is a more sane way to do this...
rm -rf **/*.pyc
rm -rf  __pychache__
rm -rf tests/__pycache__/
rm -rf src/__pycache__/

(
    set -Ceu
    cd -- "${root}"
    # The branching below differs slightly from other probcomp repos.
    if [ $# -eq 0 ]; then
        # If no args are specified, run all tests, including continuous
        # integration tests, for the selected components.
        OPENBLAS_NUM_THREADS=1 "$PYTHON" -m pytest  -vvv
    else
        # If args are specified, run all tests, but skip tests that have
        # been marked for continuous integration by using __ci_ in
        # their names.  (git grep __ci_ to find these.)
        OPENBLAS_NUM_THREADS=1 "$PYTHON" -m pytest  -k "not __ci_"  "$@"
    fi
)
