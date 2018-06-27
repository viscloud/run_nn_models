#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd tf_nets

    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    tar -xvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz

    rm faster_rcnn_resnet101_coco_2018_01_28.tar.gz
    mv faster_rcnn_resnet101_coco_2018_01_28 faster_rcnn_resnet101_coco

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
