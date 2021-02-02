import tensorflow as tf
import time


def simple_bench(runner, data_size=256):
    tic = time.time()
    runner(data_size)
    toc = time.time()
    return toc - tic


def tf_keras_model(model_name):
    # default classes=1000, include_top=True (fully-connected at the top -> doesn't have to specify input shape)
    if model_name == 'vgg16':
        model = tf.keras.applications.VGG16(weights=None,
                                            classes=1000,
                                            include_top=False,
                                            input_shape=(224, 224, 3))
    elif model_name == 'resnet50':
        model = tf.keras.applications.ResNet50(weights=None,
                                               include_top=False,
                                               input_shape=(224, 224, 3))
    elif model_name == 'mobilenet':
        model = tf.keras.applications.MobileNetV2(weights=None,
                                                  include_top=False,
                                                  input_shape=(224, 224, 3))
    elif model_name == 'inception':
        model = tf.keras.applications.InceptionV3(weights=None,
                                                  include_top=False,
                                                  input_shape=(224, 224, 3))
    else:
        raise Exception("Invalid model_name")
    return model
