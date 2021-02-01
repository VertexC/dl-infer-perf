import tensorflow as tf
import numpy as np
import time
import os, argparse

def tf2xla(sm_path, batch=1, nSteps=15):
    # default classes=1000, include_top=True (fully-connected at the top -> doesn't have to specify input shape)
    if sm_path == 'vgg16':
      model = tf.keras.applications.VGG16(weights=None, classes=1000, include_top=True)
    elif sm_path == 'resnet50':
      model = tf.keras.applications.ResNet50(weights=None)
    elif sm_path == 'mobilenet':
      model = tf.keras.applications.MobileNetV2(weights=None)
    elif sm_path == 'inception':
      model = tf.keras.applications.InceptionV3(weights=None)
    
    shape=[256,224,224,3]
    data = np.ones(shape, dtype=np.float32)
    
    avg_time=0
    for i in range(0, nSteps):
      time1 = time.time()
      ret = model.predict(data, batch_size=batch)
      time2 = time.time()
      if i < 5:
        continue
      avg_time += float(time2-time1)
      info = '-- %d, iteration time(s) is %.4f' %(i, float(time2-time1))
      print(info)

    avg_time = avg_time / (nSteps-5)
    name = os.path.basename(sm_path)
    print("@@ %s, average time(s) is %.4f" % (name, avg_time))
    print('FINISH')

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "benchmark of tf/xla")
    parser.add_argument("model", help = "tf model name")
    parser.add_argument("--xla", action='store_true', help='Flag to turn on xla')
    parser.add_argument("--device", choices=['gpu', 'cpu'], default='cpu', help='device to run on')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    arg = parser.parse_args() 

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    if arg.xla:
      tf.keras.backend.clear_session()
      tf.config.optimizer.set_jit(True) # Enable XLA
    
    if arg.device == 'cpu':
      os.environ['CUDA_VISIBLE_DEVICES'] = ''

    print(time.strftime("[localtime] %Y-%m-%d %H:%M:%S", time.localtime()))

    tf2xla(arg.model, batch=arg.batch) 