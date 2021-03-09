import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

import os.path
import time

import util


def tf2trt_runner(model_name, batch_size=1):
    # tvm cuda will have issue with mobilenet
    model, shape = util.tf_keras_model(model_name)
    model_path = model_name + '_saved_model'
    if not os.path.isdir(model_path):
        model.save(model_path)

    trt_path = model_path + '_TFTRT_FP32'

    # if not os.path.isdir(trt_path):
    # always regenerate model to avoid incompatibility between different onnx/trt version
    print('Converting to TF-TRT FP32...')
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP32,
        max_workspace_size_bytes=8000000000)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_path,
                                        conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir=trt_path)
    print('Done Converting to TF-TRT FP32')

    saved_model_loaded = tf.saved_model.load(trt_path,
                                             tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    data = np.random.rand(batch_size, *shape).astype(np.float32)
    x = tf.constant(data)

    def runner(data_size):
        for _ in range(data_size // batch_size):
            infer(x)

    return runner


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark of tf/TensorRT")
    parser.add_argument("model", help="tf model name")
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='test data size')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    runner = tf2trt_runner(args.model, batch_size=args.batch)
    throughput = util.simple_bench(runner,
                                   data_size=args.size,
                                   warmup=1,
                                   rounds=5,
                                   verbose=True)
    print(throughput)
