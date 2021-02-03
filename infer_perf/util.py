import time


def simple_bench(runner, data_size=256):
    tic = time.time()
    runner(data_size)
    toc = time.time()
    return toc - tic


def torch_model(name):
    import torchvision
    from torchvision import models
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
