import sys
#sys.path.insert(0,'/home/lc/caffe-rc5/python')
import caffe
import cv2
import numpy as np

#TensorFlow backend uses all GPU memory by default, so we need limit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

import keras2caffe

sys.path.append('/home/lc/keras2caffe/ssrnet/')

import keras_ssrnet
#import evalute_image

#converting

keras_model = keras_ssrnet.create_model(3,3,3,0.5,0.5,weights='imdb', include_top=True)

keras2caffe.convert(keras_model, 'ssrnet.prototxt', 'ssrnet.caffemodel')

#testing the model

#caffe.set_mode_cpu()
#net  = caffe.Net('ssrnet.prototxt', 'ssrnet.caffemodel', caffe.TEST)

#img = cv2.imread('man.JPG')
#img = evaluate_image.central_crop(im, 0.875)

#img = cv2.resize(img, (64, 64))
#img = img[...,::-1]  #RGB 2 BGR

#data = np.array(img, dtype=np.float32)
#data = data.transpose((2, 0, 1))
#data.shape = (1,) + data.shape

#data -= 128
#data /= 128
#data /= 256
#data -= 1.0
#data = np.divide(data, 255.0)
#data = np.subtract(data, 1.0)
#data = np.multiply(data, 2.0)

#net.blobs['data'].data[...] = data

#out = net.forward()
#preds = out['pred_a']

#classes = eval(open('class_names.txt', 'r').read())
#print("Class is: " + classes[np.argmax(preds)-1])
#print("Certainty is: " + str(preds[0][np.argmax(preds)]))
#print("pred is : "+preds)
print 'Done'


