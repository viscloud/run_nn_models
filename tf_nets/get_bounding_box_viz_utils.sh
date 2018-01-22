#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    git submodule update --init models

    if [ ! -d utils ]; then
        mkdir utils
    fi
    if [ ! -f utils/__init__.py ]; then
        touch utils/__init__.py
    fi
    cp models/research/object_detection/utils/visualization_utils.py utils
}

if [[ "$DIR" != "run_nn_models" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the run_nn_models repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi
