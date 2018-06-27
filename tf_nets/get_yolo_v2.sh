#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd tf_nets

    mkdir yolo_v2

    cd YAD2K
    wget http://pjreddie.com/media/files/yolo.weights
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg
    mv yolov2.cfg yolo.cfg

    input_width=512
    input_height=512
    sed -i "s/width=416/width=$input_width/g" yolo.cfg
    sed -i "s/height=416/height=$input_height/g" yolo.cfg

    python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
    rm yolo.weights yolo.cfg
    mv model_data/yolo.h5 ../yolo_v2

    cd $CWD

    cp -r tf_nets/YAD2K/yad2k ./
}

if [[ "$DIR" != "run_nn_models" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the run_nn_models repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi
