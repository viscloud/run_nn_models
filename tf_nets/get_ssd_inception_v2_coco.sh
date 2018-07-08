#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd tf_nets

    wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
    tar -xvf ssd_inception_v2_coco_2017_11_17.tar.gz

    rm ssd_inception_v2_coco_2017_11_17.tar.gz
    mv ssd_inception_v2_coco_2017_11_17 ssd_inception_v2_coco

    cd $CWD

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
