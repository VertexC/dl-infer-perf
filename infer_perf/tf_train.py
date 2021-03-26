import argparse
import os
import numpy as np
import datetime
import timeit
import util
import datetime
import tensorflow as tf
from tensorflow.keras import applications


def train_runner(model_name, batch_size, device='gpu', xla=True):
    if xla:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)
    else:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)
    if device == "gpu":
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Set up standard model.
    model, shape = util.tf_keras_model(model_name)
    opt = tf.optimizers.SGD(0.01)

    data = tf.random.uniform([batch_size, 224, 224, 3])
    target = tf.random.uniform([batch_size, 1],
                               minval=0,
                               maxval=999,
                               dtype=tf.int64)

    @tf.function
    def benchmark_step():
        with tf.GradientTape() as tape:
            probs = model(data, training=True)
            loss = tf.losses.sparse_categorical_crossentropy(target, probs)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    def log(s, nl=True):
        print(s, end='\n' if nl else '')

    log('Model: %s' % model_name)
    log('Batch size: %d' % batch_size)

    class runner_wrapper:
        def __init__(self, benchmark_step, batch_size):
            self.step = benchmark_step
            self.batch_size = batch_size

        def __call__(self, data_size):
            print(data_size, self.batch_size)
            for _ in range(data_size // self.batch_size):
                self.step()

    return runner_wrapper(benchmark_step, batch_size)


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
    parser.add_argument("--visual",
                        action='store_true',
                        help='Output tensorboard log for visualization')

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.visual:
        model, shape = util.tf_keras_model(args.model)
        log_dir = "logs/fit/{}/{}".format(
            args.model,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)
        x_train = tf.random.uniform([args.batch, 224, 224, 3])
        y_train = tf.random.uniform([args.batch, 1],
                                    minval=0,
                                    maxval=999,
                                    dtype=tf.int64)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x=x_train,
                  y=y_train,
                  epochs=1,
                  callbacks=[tensorboard_callback])
    else:
        runner = train_runner(args.model,
                              batch_size=args.batch,
                              xla=args.xla,
                              device=args.device)

        throughput = util.simple_bench(runner,
                                       data_size=args.size,
                                       warmup=1,
                                       rounds=5,
                                       verbose=True)
