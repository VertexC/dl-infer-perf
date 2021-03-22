import time
import os, argparse
import contextlib
import tensorflow as tf
import numpy as np
import util
import numpy as np
import traceback


@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def grappler_runner(model_name, batch_size):
    model, shape = util.tf_keras_model(model_name)
    # data = np.random.rand(batch_size, *shape).astype(np.float32)
    grap_model = tf.function(lambda x: model(x))

    class runner_wrapper:
        def __init__(self, batch_size=1):
            self.batch_size = batch_size
            self.data = np.random.rand(batch_size, *shape).astype(np.float32)

        def __call__(self, data_size):
            with options({
                    'layout_optimizer': False,
                    'function_optimization': False
            }):
                for _ in range(data_size // self.batch_size):
                    ret = grap_model(self.data)

    runner = runner_wrapper(batch_size=batch_size)
    return runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark of tf/xla")
    parser.add_argument("model", help="tf model name")
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    runner = grappler_runner(args.model, batch_size=args.batch)
    duration = util.simple_bench(runner, data_size=args.size)
    print(duration)
