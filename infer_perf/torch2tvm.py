import os
import time

import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime
import tvm.testing
import torch

import util


def torch2tvm_runner(model_name, batch_size=1, backend='cuda'):
    # TODO: add batch
    input_name = "input0"
    shape = [1, 3, 224, 224]
    data = torch.randn(shape, dtype=torch.float32)
    model = util.torch_model(model_name)

    shape_list = [(input_name, data.shape)]
    scripted_model = torch.jit.trace(model, data).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # TODO: how opt_level affects performance
    opt_level = 3
    if backend == 'llvm':
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod,
                              target='llvm',
                              target_host='llvm',
                              params=params)

        ctx = tvm.cpu()
        module = graph_runtime.GraphModule(lib["default"](ctx))
        module.set_input(input_name, data)
    else:
        target = tvm.target.cuda()
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target, params=params)

        ctx = tvm.gpu()
        module = graph_runtime.GraphModule(lib["default"](ctx))
        module.set_input(input_name, data)

    data = tvm.nd.array(data)

    def runner(data_size):
        for _ in range(data_size // batch_size):
            module.set_input(input_name, data)
            module.run()

    return runner


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark of pytorch/xla")
    parser.add_argument("model", help="pytorch model name")
    parser.add_argument("--backend",
                        choices=['cuda', 'llvm'],
                        default='cuda',
                        help='backend target to run')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')
    args = parser.parse_args()

    os.environ['TVM_BACKTRACE'] = '1'
    runner = torch2tvm_runner(args.model,
                              batch_size=args.batch,
                              backend=args.backend)
    duration = util.simple_bench(runner, args.size)
    print(duration)
