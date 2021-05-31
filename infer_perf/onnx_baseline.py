import onnxruntime as rt
import numpy as np
import util


def onnx_runner(model_name, batch_size=1):
    model_path = util.onnx_model_path(model_name)
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    input_size = 224
    x = np.random.random((1, 3, input_size, input_size)).astype(np.float32)

    def runner(data_size):
        for _ in range(data_size // batch_size):
            pred = sess.run([label_name], {input_name: x})[0]

    return runner


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark of onnx baseline")
    parser.add_argument("model", help="onnx model name")
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')

    args = parser.parse_args()

    runner = onnx_runner(args.model, batch_size=args.batch)
    throughput = util.simple_bench(runner,
                                   data_size=args.size,
                                   warmup=1,
                                   rounds=5,
                                   verbose=True)
    print(throughput)
