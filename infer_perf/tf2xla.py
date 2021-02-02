import time
import os, argparse

import tensorflow as tf
import numpy as np

import util

WARM_UP_STEPS = 5


def tf2xla_runner(model_name, batch_size=1, xla=False, device='gpu'):
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
    model = util.tf_keras_model(model_name)
    data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)

    def runner(data_size):
        for _ in range(data_size // batch_size):
            ret = model.predict(data, batch_size=batch_size)

    return runner


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
    arg = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    runner = tf2xla_runner(arg.model,
                           batch_size=arg.batch,
                           xla=arg.xla,
                           device=arg.device)
    duration = util.simple_bench(runner, 256)
    print(duration)
