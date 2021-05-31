import time
import os
import numpy as np


def simple_bench(runner, data_size=256, warmup=0, rounds=1, sleep=0, verbose=False):
    for i in range(warmup):
        tic = time.time()
        runner(data_size)
        toc = time.time()
        duration = toc - tic
        throughput = data_size / duration
        if verbose:
            print('warmup {}: throughput {} imgs/s'.format(i, throughput))
            print('warmup {}: time {} us'.format(i, duration*1000000))
        if sleep > 0:
            print('sleeping :{}s'.format(sleep))
            time.sleep(sleep)
    throughputs = []
    durations = []
    for i in range(rounds):
        tic = time.time()
        runner(data_size)
        toc = time.time()
        duration = toc - tic
        throughput = data_size / duration
        throughputs.append(throughput)
        durations.append(duration)
        if verbose:
            print('round {}: throughput {} imgs/s'.format(i, throughput))
            print('round {}: time {} us'.format(i, duration*1000000))
        if sleep > 0:
            print('sleeping :{}s'.format(sleep))
            time.sleep(sleep)
    print(throughputs)
    print(durations)
    throughput_avg = np.mean(throughputs)
    throughput_std = np.std(throughputs)
    print('throughput: {}+/-{}'.format(throughput_avg, throughput_std))
    duration_avg = np.mean(durations)
    duration_std = np.std(durations)
    print('durations: {}+/-{}'.format(duration_avg, duration_std))
    return throughput_avg


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


def tf_vgg_small(input_shape=(224, 224, 3), classes=1000):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.python.keras.engine import training
    img_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    # Create model
    model = training.Model(img_input, x, name='vgg-small')
    return model


def tf_keras_model(name):
    import tensorflow as tf
    # set include_top to False and input_shape explicitly as mobilenet has different default input shape
    shape = [224, 224, 3]
    if name == 'vgg16':
        model = tf.keras.applications.VGG16(weights=None,
                                            classes=1000,
                                            include_top=True,
                                            input_shape=(224, 224, 3))
    elif name == 'resnet50':
        model = tf.keras.applications.ResNet50(weights=None,
                                               include_top=True,
                                               input_shape=(224, 224, 3))
    elif name == 'mobilenet':
        model = tf.keras.applications.MobileNetV2(weights=None,
                                                  include_top=True,
                                                  input_shape=(224, 224, 3))
    elif name == 'inception':
        model = tf.keras.applications.InceptionV3(weights=None,
                                                  include_top=True,
                                                  input_shape=(224, 224, 3))
    elif name == 'vgg-small':
        model = tf_vgg_small()
    else:
        raise Exception("Invalid tf model name")
    return model, shape
