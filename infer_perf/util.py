import time
import os
import numpy as np


def simple_bench(runner, data_size=256, warmup=0, rounds=1, verbose=False):
    for i in range(warmup):
        tic = time.time()
        runner(data_size)
        toc = time.time()
        throughput = data_size / (toc - tic)
        if verbose:
            print('warmup {}: throughput {} imgs/s'.format(i, throughput))
    throughputs = []
    for i in range(rounds):
        tic = time.time()
        runner(data_size)
        toc = time.time()
        throughput = data_size / (toc - tic)
        throughputs.append(throughput)
        if verbose:
            print('round {}: throughput {} imgs/s'.format(i, throughput))
    avg = np.mean(throughputs)
    std = np.std(throughputs)
    print('throughput: {}+/-{}'.format(avg, std))
    return avg


def memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')


def onnx_model(name):
    import onnx
    shape = [3, 224, 224]
    model_path = onnx_model_path(name)
    model = onnx.load(model_path)
    return model, shape


def onnx_model_path(name):
    if name not in ['resnet50', 'mobilenet', 'vgg16', 'inception']:
        raise Exception("Invalid onnx model name")
    return 'onnx_models/{}.onnx'.format(name)


def torch_model(name):
    shape = [3, 224, 224]
    import torchvision
    if name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False, progress=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, progress=True)
    elif name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=False,
                                                progress=True)
    elif name == 'inception':
        model = torchvision.models.inception_v3(pretrained=False,
                                                progress=True)
        shape = [3, 299, 299]
    else:
        raise Exception("Invalid pytorch model name")
    return model, shape


def tf_keras_model(name):
    import tensorflow as tf
    # set include_top to False and input_shape explicitly as mobilenet has different default input shape
    shape = [224, 224, 3]
    if name == 'vgg16':
        model = tf.keras.applications.VGG16(weights=None,
                                            classes=1000,
                                            include_top=False,
                                            input_shape=(224, 224, 3))
    elif name == 'resnet50':
        model = tf.keras.applications.ResNet50(weights=None,
                                               include_top=False,
                                               input_shape=(224, 224, 3))
    elif name == 'mobilenet':
        model = tf.keras.applications.MobileNetV2(weights=None,
                                                  include_top=False,
                                                  input_shape=(224, 224, 3))
    elif name == 'inception':
        model = tf.keras.applications.InceptionV3(weights=None,
                                                  include_top=False,
                                                  input_shape=(299, 299, 3))
        shape = [299, 299, 3]
    else:
        raise Exception("Invalid tf model name")
    return model, shape
