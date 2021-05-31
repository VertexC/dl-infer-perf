def trt_runner(fe, model_name, batch_size, device):
    if device == 'cpu':
        return None
    if fe == 'tf':
        from tf2trt import tf2trt_runner
        return tf2trt_runner(model_name, batch_size=batch_size)
    elif fe == 'onnx':
        from onnx2trt import onnx2trt_runner
        return onnx2trt_runner(model_name, batch_size=batch_size)
    return None
