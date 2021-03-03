import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

import util

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
"""
A few notes regards to the build engine

"""
# def build_engine(model_path):
#     with trt.Builder(TRT_LOGGER) as builder, \
#         builder.create_network() as network, \
#         trt.OnnxParser(network, TRT_LOGGER) as parser:
#         builder.max_workspace_size = 1 << 20
#         builder.max_batch_size = 1
#         with open(model_path, "rb") as f:
#             parser.parse(f.read())
#         engine = builder.build_cuda_engine(network)
#         return engine


def build_engine(model_path):
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        builder.max_workspace_size = 1 << 20
        builder.max_batch_size = 1

        # # add below two linse to avoid `[TensorRT] ERROR: Network must have at least one output`
        # last_layer = network.get_layer(network.num_layears - 1)
        # network.mark_output(last_layer.get_output(0))

        engine = builder.build_cuda_engine(network)
        return engine


def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu


def onnx2trt_runner(model, batch_size=1):
    model_path = util.onnx_model_path(model)
    input_size = 224
    inputs = np.random.random(
        (1, 3, input_size, input_size)).astype(np.float32)

    engine = build_engine(model_path)
    context = engine.create_execution_context()

    in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)

    def runner(data_size):
        for _ in range(data_size // batch_size):
            inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu,
                      out_gpu, stream)

    return runner


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark of onnx/trt")
    parser.add_argument("model", help="onnx model name")
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')
    args = parser.parse_args()

    runner = onnx2trt_runner(args.model, batch_size=args.batch)
    duration = util.simple_bench(runner, args.size)
    print(duration)
