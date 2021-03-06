import time
import os, argparse

import tensorflow as tf
import numpy as np

import util

print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


def xla_runner(fe, model_name, batch_size, device, xla, test=False):
    if fe != 'tf':
        return None
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

    class runner_wrapper:
        def __init__(self, graph_model, need_eval, batch_size=1):
            self.batch_size = batch_size
            self.data = np.random.rand(batch_size, *shape).astype(np.float32)
            self.need_eval = need_eval
            self.graph_model = graph_model

        def __call__(self, data_size):
            if self.need_eval:
                self.session_runner(data_size)
            else:
                for _ in range(data_size // self.batch_size):
                    ret = self.graph_model(self.data)

        # explicitly eval is only needed when eager_execution is off
        def session_runner(self, data_size):
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                for _ in range(data_size // self.batch_size):
                    ret = self.graph_model(self.data)
                    ret_np = ret.eval()

    graf_model = tf.function(lambda x: model(x))

    runner = runner_wrapper(graf_model, False, batch_size=batch_size)
    return runner


def xla(model_name, batch_size=1, device='gpu', xla=True, test=False):
    return xla_runner('tf', model_name, batch_size, device, xla, test=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark of tf/xla")
    parser.add_argument("model", help="tf model name")
    parser.add_argument("--xla",
                        action='store_true',
                        help='Flag to turn on xla')
    parser.add_argument("--device",
                        choices=['gpu', 'cpu'],
                        default='gpu',
                        help='device to run on')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')
    parser.add_argument("--test",
                        action='store_true',
                        help='Store temporary result')

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    runner = xla(args.model,
                 batch_size=args.batch,
                 xla=args.xla,
                 device=args.device,
                 test=args.test)
    throughput = util.simple_bench(runner,
                                   data_size=args.size,
                                   warmup=1,
                                   rounds=5,
                                   verbose=True)
    print(throughput)
