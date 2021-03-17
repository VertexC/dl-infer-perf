import tvm
from tvm import te
from tvm import relay
from tvm.contrib import graph_runtime

import numpy as np
import tensorflow as tf

import os.path
import time

import util


def tf2tvm_runner(model_name, batch_size=1, backend='cuda'):
    # tvm cuda will have issue with mobilenet
    if model_name == 'mobilenet' and backend == 'cuda':
        return None
    model, shape = util.tf_keras_model(model_name)
    # TODO: why tvm needs reversed shape
    shape = shape[::-1]
    data = np.random.rand(batch_size, *shape)
    # input_name has to match model's input name
    # use  model.input_names[0] instead of input_1 to compile different models inside same round
    # TODO: why would same models with cuda/lvvm can compile in same process? (different backends models doens't affect each other?)
    input_name = model.input_names[0]
    shape_dict = {input_name: data.shape}
    mod, params = relay.frontend.from_keras(model, shape_dict)

    if backend == 'llvm':
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod,
                              target='llvm',
                              target_host='llvm',
                              params=params)

        ctx = tvm.cpu()
        module = graph_runtime.GraphModule(lib["default"](ctx))
    else:
        with tvm.transform.PassContext(opt_level=3):
            # has to specify target to tvm.target.cuda(), 'cuda' doesn't work
            lib = relay.build(mod, target=tvm.target.cuda(), params=params)

        ctx = tvm.gpu()
        module = graph_runtime.GraphModule(lib["default"](ctx))

    # FIXME: why neccessary to have dtype as float32 here, failed with float64?
    dtype = "float32"
    data = tvm.nd.array(data.astype(dtype))

    def runner(data_size):
        for _ in range(data_size // batch_size):
            module.set_input(input_name, data)
            tvm_output = module.get_output(0)

    return runner


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark of tf/xla")
    parser.add_argument("model", help="tf model name")
    parser.add_argument("--backend",
                        choices=['cuda', 'llvm'],
                        default='cuda',
                        help='backend target to run')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')
    args = parser.parse_args()

    os.environ['TVM_BACKTRACE'] = '1'
    runner = tf2tvm_runner(args.model,
                           batch_size=args.batch,
                           backend=args.backend)
    duration = util.simple_bench(runner, args.size)
    print(duration)
