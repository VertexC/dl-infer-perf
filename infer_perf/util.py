import tensorflow as tf


def tf_keras_model(model_name):
    # default classes=1000, include_top=True (fully-connected at the top -> doesn't have to specify input shape)
    if model_name == 'vgg16':
        model = tf.keras.applications.VGG16(
            weights=None, classes=1000, include_top=True)
    elif model_name == 'resnet50':
        model = tf.keras.applications.ResNet50(weights=None)
    elif model_name == 'mobilenet':
        model = tf.keras.applications.MobileNetV2(weights=None)
    elif model_name == 'inception':
        model = tf.keras.applications.InceptionV3(weights=None)
    else:
        raise ("Invalid model_name")
    return model
