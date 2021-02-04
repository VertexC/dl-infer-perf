import os
import time

import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib import graph_runtime

import util


def onnx2tvm_runner(model_name, batch_size=1, backend='cuda'):
    model, shape = util.onnx_model(model_name)

    data = np.random.rand(batch_size, *shape).astype(np.float32)
    input_name = model.graph.input[0].name

    shape_dict = {input_name: data.shape}
    mod, params = relay.frontend.from_onnx(model, shape_dict)

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
    parser.add_argument("model", help="onnx model name")
    parser.add_argument("--backend",
                        choices=['cuda', 'llvm'],
                        default='cuda',
                        help='backend target to run')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')
    args = parser.parse_args()

    os.environ['TVM_BACKTRACE'] = '1'
    runner = onnx2tvm_runner(args.model,
                             batch_size=args.batch,
                             backend=args.backend)
    duration = util.simple_bench(runner, args.size)
    print(duration)
