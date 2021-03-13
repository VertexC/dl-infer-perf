def tvm_runner(fe, model_name, batch_size, device):
    backend = 'cuda'
    if device == 'cpu':
        backend = 'llvm'
    if fe == 'torch':
        from torch2tvm import torch2tvm_runner
        return torch2tvm_runner(model_name,
                                batch_size=batch_size,
                                backend=backend)
    elif fe == 'tf':
        from tf2tvm import tf2tvm_runner
        return tf2tvm_runner(model_name,
                             batch_size=batch_size,
                             backend=backend)
    elif fe == 'onnx':
        from onnx2tvm import onnx2tvm_runner
        return onnx2tvm_runner(model_name,
                               batch_size=batch_size,
                               backend=backend)

    return None
