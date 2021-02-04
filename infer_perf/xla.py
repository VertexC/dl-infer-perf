import time
import os, argparse

import tensorflow as tf
import numpy as np

import util


def xla_runner(fe, model_name, batch_size, device, xla):
    if fe != 'tf':
        return None
    if xla:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)    # Enable XLA
    else:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    model, shape = util.tf_keras_model(model_name)
    data = np.random.rand(batch_size, *shape).astype(np.float32)

    def runner(data_size):
        for _ in range(data_size // batch_size):
            ret = model.predict(data, batch_size=batch_size)

    return runner


def xla(model_name, batch_size=1, device='gpu', xla=True):
    xla_runner('tf', model_name, batch_size, device, xla)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark of tf/xla")
    parser.add_argument("model", help="tf model name")
    parser.add_argument("--xla",
                        action='store_true',
                        help='Flag to turn on xla')
    parser.add_argument("--device",
                        choices=['gpu', 'cpu'],
                        default='cpu',
                        help='device to run on')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    runner = xla(args.model,
                 batch_size=args.batch,
                 xla=args.xla,
                 device=args.device)
    duration = util.simple_bench(runner, 256)
    print(duration)
