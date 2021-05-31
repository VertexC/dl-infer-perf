import time
import os, argparse
import datetime
from datetime import datetime

import tensorflow as tf
import numpy as np

import util

print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, False)

if gpus:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

def xla_runner(fe, model_name, batch_size, device, xla):
    if fe != 'tf':
        return None
   
    print('TF inter thread: {}, intra thread: {}'.format(
        tf.config.threading.get_inter_op_parallelism_threads(),
        tf.config.threading.get_intra_op_parallelism_threads()))
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
                    # import pdb; pdb.set_trace()
                    print("graph start", datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f"))
                    start = time.time()
                    ret = self.graph_model(self.data)
                    end = time.time()
                    print("graph end", datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f"))
                    print("graph_model time: %s us" % ((end - start)*10**6))
                    # print(ret)
                    ret.numpy()

        # explicitly eval is only needed when eager_execution is off
        def session_runner(self, data_size):
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                for _ in range(data_size // self.batch_size):
                    ret = self.graph_model(self.data)
                    ret_np = ret.eval()

    graph_mode = tf.function(lambda x: model(x))

    runner = runner_wrapper(graph_mode, False, batch_size=batch_size)
    return runner


def xla(model_name, batch_size=1, device='gpu', xla=True):
    return xla_runner('tf',
                      model_name,
                      batch_size,
                      device,
                      xla)


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
                        type=str,
                        default="",
                        help='Output tensorboard log for visualization')
    parser.add_argument("-w",
                        "--warmup",
                        default=3,
                        type=int,
                        help="warm up rounds")
    parser.add_argument("-r",
                        "--rounds",
                        default=30,
                        type=int,
                        help="rounds to execute runner")
    parser.add_argument("--profile",
                        type=str,
                        default="",
                        help='Add profile logs')
    parser.add_argument("--threads",
                        type=int,
                        default=0,
                        help='inter_op_parallelism_threads')
    parser.add_argument("--log",
                        type=str,
                        default='3',
                        help='TF_CPP_MIN_VLOG_LEVEL')
    parser.add_argument("--sleep",
                        type=int,
                        default=0,
                        help='time to sleep between rounds')

    args = parser.parse_args()

    # os.environ['TF_CPP_MIN_VLOG_LEVEL'] = args.log

    log_dir = "logs/infer/{}".format(
            args.model)

    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.visual != "":
        model, shape = util.tf_keras_model(args.model)
        
        log_dir = os.path.join(log_dir, args.visual)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)
        x = np.random.rand(args.batch, *shape).astype(np.float32)
        model.predict(x, callbacks=[tensorboard_callback])

    
    if args.profile != "":
        # from tensorflow_core.python.eager import profiler

        log_dir = os.path.join(log_dir, args.profile)
        # profiler = tf.profiler.experimental.Profile(log_dir)
        runner = xla(args.model,
                    batch_size=args.batch,
                    xla=args.xla,
                    device=args.device)

        util.simple_bench(runner,
                        data_size=args.size,
                        warmup=args.warmup,
                        rounds=args.rounds,
                        verbose=True)
        
        tf.profiler.experimental.start(log_dir)
        throughput = util.simple_bench(runner,
                                       data_size=args.size,
                                       warmup=args.warmup,
                                       rounds=args.rounds,
                                       verbose=True)
        tf.profiler.experimental.stop()
    else:
        runner = xla(args.model,
                     batch_size=args.batch,
                     xla=args.xla,
                     device=args.device)
        throughput = util.simple_bench(runner,
                                       data_size=args.size,
                                       warmup=args.warmup,
                                       rounds=args.rounds,
                                       verbose=True,
                                       sleep=args.sleep)
        print(throughput)
