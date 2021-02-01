import time
import os, argparse

import tensorflow as tf
import numpy as np

from util import tf_keras_model

WARM_UP_STEPS = 5


def tf2xla_runner(model_name, batch_size=1, nSteps=15):
    model = tf_keras_model(model_name)

    shape = [256, 224, 224, 3]
    data = np.ones(shape, dtype=np.float32)

    avg_time = 0
    for i in range(0, nSteps):
        tic = time.time()
        ret = model.predict(data, batch_size=batch_size)
        toc = time.time()
        # a warm up process
        if i < WARM_UP_STEPS:
            continue
        avg_time += float(toc - tic)
        info = '-- %d, iteration time(s) is %.4f' % (i, float(toc - tic))
        print(info)

    avg_time = avg_time / (nSteps - WARM_UP_STEPS)
    print("@@ %s, average time(s) is %.4f" % (model, avg_time))
    print('FINISH')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark of tf/xla")
    parser.add_argument("model", help="tf model name")
    parser.add_argument(
        "--xla", action='store_true', help='Flag to turn on xla')
    parser.add_argument(
        "--device",
        choices=['gpu', 'cpu'],
        default='cpu',
        help='device to run on')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    arg = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if arg.xla:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)    # Enable XLA

    if arg.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

    tf2xla_runner(arg.model, batch_size=arg.batch)
