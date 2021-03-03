import time
import os, argparse

import tensorflow as tf
import numpy as np

import util


def xla_runner(fe, model_name, batch_size, device, xla, eager):
    assert not (xla and eager)

    if fe != 'tf':
        return None
    if not eager:
        tf.compat.v1.disable_eager_execution()
    if xla:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)
    else:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)
    if device == 'cpu':
        # FIXME: add a better way to use cpu device in tf
        # os.environ['CUDA_VISIBLE_DEVICES'] = ''
        pass
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    model, shape = util.tf_keras_model(model_name)
    # data = np.random.rand(batch_size, *shape).astype(np.float32)
    if xla:
        grap_model = tf.function(lambda x: model(x))
    else:
        grap_model = model.predict

    class runner_wrapper:
        def __init__(self, batch_size=1):
            self.batch_size = batch_size
            self.data = np.random.rand(batch_size, *shape).astype(np.float32)

        def __call__(self, data_size):
            for _ in range(data_size // self.batch_size):
                ret = grap_model(self.data)

    runner = runner_wrapper(batch_size=256)
    return runner


def xla(model_name, batch_size=1, device='gpu', xla=True, eager=False):
    return xla_runner('tf', model_name, batch_size, device, xla, eager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark of tf/xla")
    parser.add_argument("model", help="tf model name")
    parser.add_argument("--xla",
                        action='store_true',
                        help='Flag to turn on xla')
    parser.add_argument("--eager",
                        action='store_true',
                        help='Flat to trun on eager execution')
    parser.add_argument("--device",
                        choices=['gpu', 'cpu'],
                        default='gpu',
                        help='device to run on')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    runner = xla(args.model,
                 batch_size=args.batch,
                 xla=args.xla,
                 eager=args.eager,
                 device=args.device)
    duration = util.simple_bench(runner, data_size=args.size)
    print(duration)
