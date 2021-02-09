import time
import os


def simple_bench(runner, data_size=256):
    tic = time.time()
    runner(data_size)
    toc = time.time()
    return toc - tic


def memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')


def onnx_model(name):
    import onnx
    shape = [3, 224, 224]
    if name == 'resnet50':
        model = onnx.load("onnx_models/resnet50.onnx")
    elif name == 'mobilenet':
        model = onnx.load("onnx_models/mobilenet.onnx")
    elif name == 'vgg16':
        model = onnx.load("onnx_models/vgg16.onnx")
    elif name == 'inception':
        model = onnx.load("onnx_models/inception.onnx")
    else:
        raise Exception("Invalid onnx model name")
    return model, shape


def onnx_model_path(name):
    return 'onnx_models/{}.onnx'.format(name)


def torch_model(name):
    shape = [3, 224, 224]
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
                                                  input_shape=(299, 299, 3))
        shape = [299, 299, 3]
    else:
        raise Exception("Invalid tf model name")
    return model, shape
