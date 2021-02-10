#!/bin/bash
SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")

# change workspace
cd $DIR

if [ $# == 2 ]
then
    JOB=$1
    OPTIMIZER=$2
    echo "Start to run job $JOB $OPTIMIZER"
else
    echo "Missing argument for job name"
    exit 1
fi

JOB_DIR="${DIR}/${JOB}"
if [ -d $JOB_DIR ]
then 
    echo "Dir $JOB_DIR already exists"
else
    mkdir $JOB_DIR
fi

# run jobs
if [ $OPTIMIZER == "tvm" ]
then
    source ../tvm-tf-env/bin/activate && 
        python3 infer_perf/executor.py tf2tvm.json ${JOB_DIR}/tf2tvm.csv && 
        deactivate

    source ../tvm-onnx-env/bin/activate && 
        python3 infer_perf/executor.py onnx2tvm.json ${JOB_DIR}/onnx2tvm.csv && 
        deactivate

    source ../tvm-torch-env/bin/activate && 
        python3 infer_perf/executor.py torch2tvm.json ${JOB_DIR}/torch2tvm.csv && 
        deactivate
elif [  $OPTIMIZER == "xla" ]
then
    python3 infer_perf/executor.py xla.json ${JOB_DIR}/tf2xla.csv
elif [ $OPTIMIZER == "trt" ]
then 
    source ../trt-tf-env/bin/activate && 
        python3 infer_perf/executor.py tf2trt.json ${JOB_DIR}/tf2trt.csv &&
        deactivate

    python3 infer_perf/executor.py onnx2trt.json ${JOB_DIR}/onnx2trt.csv
fi

# xla

# trt