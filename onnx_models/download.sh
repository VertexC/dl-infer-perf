#!/bin/bash
SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")
echo $DIR
if [ ! -f ${DIR}/resnet50.onnx ]; then
    wget -O ${DIR}/resnet50.onnx https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx
fi
if [ ! -f ${DIR}/mobilenet.onnx ]; then
    wget -O ${DIR}/mobilenet.onnx https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
fi
if [ ! -f ${DIR}/vgg16.onnx ]; then
    wget -O ${DIR}/vgg16.onnx https://github.com/onnx/models/blob/master/vision/classification/vgg/model/vgg16-7.onnx
fi
if [ ! -f ${DIR}/inception.onnx ]; then
    wget -O ${DIR}/inception.onnx https://github.com/onnx/models/blob/master/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx
fi
