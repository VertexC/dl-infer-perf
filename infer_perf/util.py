import time
import os, psutil


def simple_bench(runner, data_size=256):
    tic = time.time()
    runner(data_size)
    toc = time.time()
    return toc - tic


def memory_usage():
    process = psutil.Process(os.getpid())
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')


def onnx_model(name):
    import onnx
    if name == 'resnet50':
        model = onnx.load("onnx_models/resnet50-caffe2-v1-9.onnx")
    else:
        raise Exception("Invalid onnx model name")
    return model


def torch_model(name):
    import torchvision
    if name == 'vgg16':
        model = torchvision.models.resnet50(pretrained=False, progress=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False, progress=True)
    elif name == 'mobilenet':
        model = torchvision.models.vgg16(pretrained=False, progress=True)
    elif name == 'inception':
        model = torchvision.models.inception_v3(pretrained=False,
                                                progress=True)
    else:
        raise Exception("Invalid pytorch model name")
    return model


def tf_keras_model(name):
    import tensorflow as tf
    # set include_top to False and input_shape explicitly as mobilenet has different default input shape
    if name == 'vgg16':
        model = tf.keras.applications.VGG16(weights=None,
                                            classes=1000,
                                            include_top=False,
                                            input_shape=(224, 224, 3))
        # print("generated", model)
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
                                                  input_shape=(224, 224, 3))
    else:
        raise Exception("Invalid tf model name")
    return model
