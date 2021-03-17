# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import numpy as np
import timeit
import util

import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras import applications
from tensorflow.keras.mixed_precision import experimental as mixed_precision
# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=100,
                    help='number of benchmark iterations')

def train_runner(model_name,
                 batch_size,
                 device='gpu',
                 xla=True):
    if xla:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)
    else:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)
    # Horovod: initialize Horovod.
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    if device == "gpu":
        gpus = tf.config.experimental.list_physical_devices('GPU')
        #for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    # Set up standard model.
    # model = getattr(applications, "ResNet50")(weights=None)
    model, shape = util.tf_keras_model(model_name)
    opt = tf.optimizers.SGD(0.01)

    data = tf.random.uniform([batch_size, 224, 224, 3])
    target = tf.random.uniform([batch_size, 1], minval=0, maxval=999, dtype=tf.int64)


    @tf.function
    def benchmark_step(first_batch=False):
        # Horovod: use DistributedGradientTape
        with tf.GradientTape() as tape:
            probs = model(data, training=True)
            loss = tf.losses.sparse_categorical_crossentropy(target, probs)

        # Horovod: add Horovod Distributed GradientTape.

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

    def log(s, nl=True):
        if hvd.rank() != 0:
            return
        print(s, end='\n' if nl else '')


    log('Model: %s' % model_name)
    log('Batch size: %d' % batch_size)
    log('Number of %ss: %d' % (device, hvd.size()))

    class runner_wrapper:
        def __init__(self, benchmark_step, batch_size):
            self.first_batch = False
            self.step = benchmark_step
            self.batch_size = batch_size

        def __call__(self, data_size):
            if self.first_batch:
                self.first_batch = False
                self.step(first_batch=True)
            else:
                for _ in range(data_size // self.batch_size):
                    self.step(first_batch=False)

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
    parser.add_argument("--test",
                        action='store_true',
                        help='Store temporary result')

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set up logging for tensorboard
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/func/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)

   
    runner = train_runner(args.model,
                 batch_size=args.batch,
                 xla=args.xla,
                 device=args.device)

    throughput = util.simple_bench(runner,
                                   data_size=args.size,
                                   warmup=1,
                                   rounds=5,
                                   verbose=True)
