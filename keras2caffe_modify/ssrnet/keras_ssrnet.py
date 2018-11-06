'''
Copyright 2017 TensorFlow Authors and Kent Sommer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import numpy as np

# Sys
import warnings
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation,Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras import regularizers
from keras import initializers
from keras.models import Model
# Backend
from keras import backend as K
# Utils
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file


#########################################################################################
# Implements the Inception Network v4 (http://arxiv.org/pdf/1602.07261v1.pdf) in Keras. #
#########################################################################################

WEIGHTS_PATH = '/home/lc/keras2caffe/ssrnet/ssrnet_3_3_3_64_0.5_0.5.h5'
WEIGHTS_PATH_NO_TOP = '/home/lc/keras2caffe/ssrnet/ssrnet_3_3_3_64_0.25_0.75.h5'


def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x

def merge_age(x,s1,s2,s3,lambda_local,lambda_d):
            a = x[0][:,0]*0
            b = x[0][:,0]*0
            c = x[0][:,0]*0
            A = s1*s2*s3
            V = 101

            for i in range(0,s1):
                a = a+(i+lambda_local*x[6][:,i])*x[0][:,i]
            a = K.expand_dims(a,-1)
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j+lambda_local*x[7][:,j])*x[1][:,j]
            b = K.expand_dims(b,-1)
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k+lambda_local*x[8][:,k])*x[2][:,k]
            c = K.expand_dims(c,-1)
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))
            age = (a+b+c)*V
            return age


def ssrnet_base(inputs,s1,s2,s3,lambda_local,lambda_d):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 64 x 64 x 3 (th) or 3 x 64 x 64 (th)
    #inputs = Input(shape=self._input_shape,name='input_1')

    #-------------------------------------------------------------------------------------------------------------------------
    x = Conv2D(32,(3,3),name='conv2d_1')(inputs)
    x = BatchNormalization(axis=-1,name='batch_normlization_1')(x)
    x = Activation('relu',name='activation_1')(x)
    x_layer1 = AveragePooling2D(2,2,padding='same',name='average_pooling2d_1')(x)
    x = Conv2D(32,(3,3),name='conv2d_2')(x_layer1)
    x = BatchNormalization(axis=-1,name='batch_normlization_2')(x)
    x = Activation('relu',name='activation_2')(x)
    x_layer2 = AveragePooling2D(2,2,padding='same',name='average_pooling2d_2')(x)
    x = Conv2D(32,(3,3),name='conv2d_3')(x_layer2)
    x = BatchNormalization(axis=-1,name='batch_normlization_3')(x)
    x = Activation('relu',name='activation_3')(x)
    x_layer3 = AveragePooling2D(2,2,padding='same',name='average_pooling2d_3')(x)
    x = Conv2D(32,(3,3),name='conv2d_4')(x_layer3)
    x = BatchNormalization(axis=-1,name='batch_normlization_4')(x)
    x = Activation('relu',name='activation_4')(x)
    #-------------------------------------------------------------------------------------------------------------------------
    s = Conv2D(16,(3,3),name='con2d_5')(inputs)
    s = BatchNormalization(axis=-1,name='batch_normlization_5')(s)
    s = Activation('tanh',name='activation_5')(s)
    s_layer1 = MaxPooling2D(2,2,padding='same',name='max_pooling2d_1')(s)
    s = Conv2D(16,(3,3),name='conv2d_6')(s_layer1)
    s = BatchNormalization(axis=-1,name='batch_normlization_6')(s)
    s = Activation('tanh',name='activation_6')(s)
    s_layer2 = MaxPooling2D(2,2,padding='same',name='max_pooling2d_2')(s)
    s = Conv2D(16,(3,3),name='conv2d_7')(s_layer2)
    s = BatchNormalization(axis=-1,name='batch_normlization_7')(s)
    s = Activation('tanh',name='activation_7')(s)
    s_layer3 = MaxPooling2D(2,2,padding='same',name='max_pooling2d_3')(s)
    s = Conv2D(16,(3,3),name='conv2d_8')(s_layer3)
    s = BatchNormalization(axis=-1,name='batch_normlization_8')(s)
    s = Activation('tanh',name='activation_8')(s)
    
    #-------------------------------------------------------------------------------------------------------------------------
    # Classifier block
    s_layer4 = Conv2D(10,(1,1),activation='relu',name='conv2d_9')(s)
    s_layer4 = Flatten(name='flatten_1')(s_layer4)
    s_layer4_mix = Dropout(0.2,name='dropout_1')(s_layer4)
    s_layer4_mix = Dense(units=s1, activation="relu",name='dense_1')(s_layer4_mix)
    
    x_layer4 = Conv2D(10,(1,1),activation='relu',name='conv2d_10')(x)
    x_layer4 = Flatten(name='flatten_2')(x_layer4)
    x_layer4_mix = Dropout(0.2,name='dropout_2')(x_layer4)
    x_layer4_mix = Dense(units=s1, activation="relu",name='dense_2')(x_layer4_mix)
    
    feat_a_s1_pre = Multiply(name='multiply_1')([s_layer4,x_layer4])
    delta_s1 = Dense(1,activation='tanh',name='delta_s1')(feat_a_s1_pre)
        
    feat_a_s1 = Multiply(name='multiply_2')([s_layer4_mix,x_layer4_mix])
    feat_a_s1 = Dense(2*s1,activation='relu',name='dense_3')(feat_a_s1)
    pred_a_s1 = Dense(units=s1, activation="relu",name='pred_stage1')(feat_a_s1)
    
    local_s1 = Dense(units=s1, activation='tanh', name='local_delta_stage1')(feat_a_s1)
    #-------------------------------------------------------------------------------------------------------------------------
    s_layer2 = Conv2D(10,(1,1),activation='relu',name='conv2d_11')(s_layer2)
    s_layer2 = MaxPooling2D(4,4,padding='same',name='max_pooling2d_4')(s_layer2)
    s_layer2 = Flatten(name='flatten_3')(s_layer2)
    s_layer2_mix = Dropout(0.2,name='dropout_3')(s_layer2)
    s_layer2_mix = Dense(s2,activation='relu',name='dense_4')(s_layer2_mix)
    
    x_layer2 = Conv2D(10,(1,1),activation='relu',name='conv2d_12')(x_layer2)
    x_layer2 = AveragePooling2D(4,4,padding='same',name='average_pooling2d_4')(x_layer2)
    x_layer2 = Flatten(name='flatten_4')(x_layer2)
    x_layer2_mix = Dropout(0.2,name='dropout_4')(x_layer2)
    x_layer2_mix = Dense(s2,activation='relu',name='dense_5')(x_layer2_mix)
    
    feat_a_s2_pre = Multiply(name='multiply_3')([s_layer2,x_layer2])
    delta_s2 = Dense(1,activation='tanh',name='delta_s2')(feat_a_s2_pre)
        
    feat_a_s2 = Multiply(name='multiply_4')([s_layer2_mix,x_layer2_mix])
    feat_a_s2 = Dense(2*s2,activation='relu',name='dense_6')(feat_a_s2)
    pred_a_s2 = Dense(units=s2, activation="relu",name='pred_stage2')(feat_a_s2)
    
    local_s2 = Dense(units=s2, activation='tanh', name='local_delta_stage2')(feat_a_s2)
    #-------------------------------------------------------------------------------------------------------------------------
    s_layer1 = Conv2D(10,(1,1),activation='relu',name='conv2d_13')(s_layer1)
    s_layer1 = MaxPooling2D(8,8,padding='same',name='max_pooling2d_5')(s_layer1)
    s_layer1 = Flatten(name='flatten_5')(s_layer1)
    s_layer1_mix = Dropout(0.2,name='dropout_5')(s_layer1)
    s_layer1_mix = Dense(s3,activation='relu',name='dense_7')(s_layer1_mix)
    
    x_layer1 = Conv2D(10,(1,1),activation='relu',name='conv2d_14')(x_layer1)
    x_layer1 = AveragePooling2D(8,8,padding='same',name='average_pooling2d_5')(x_layer1)
    x_layer1 = Flatten(name='flatten_6')(x_layer1)
    x_layer1_mix = Dropout(0.2,name='dropout_6')(x_layer1)
    x_layer1_mix = Dense(s3,activation='relu',name='dense_8')(x_layer1_mix)

    feat_a_s3_pre = Multiply(name='multiply_5')([s_layer1,x_layer1])
    delta_s3 = Dense(1,activation='tanh',name='delta_s3')(feat_a_s3_pre)
        
    feat_a_s3 = Multiply(name='multiply_6')([s_layer1_mix,x_layer1_mix])
    feat_a_s3 = Dense(2*s3,activation='relu',name='dense_9')(feat_a_s3)
    pred_a_s3 = Dense(units=s3, activation="relu",name='pred_stage3')(feat_a_s3)
    
    local_s3 = Dense(units=s3, activation='tanh', name='local_delta_stage3')(feat_a_s3)

    #pred_a = Lambda(merge_age,arguments={'s1':s1,'s2':s2,'s3':s3,'lambda_local':lambda_local,'lambda_d':lambda_d},output_shape=(1,),name='pred_a')([pred_a_s1,pred_a_s2,pred_a_s3,delta_s1,delta_s2,delta_s3, local_s1, local_s2, local_s3])
    return [pred_a_s1,pred_a_s2,pred_a_s3,delta_s1,delta_s2,delta_s3, local_s1, local_s2, local_s3]

def ssrnet(s1,s2,s3,lambda_local,lambda_d, weights, include_top):
    '''
    Creates the inception v4 network

    Args:
    	num_classes: number of classes
    	dropout_keep_prob: float, the fraction to keep before final layer.
    
    Returns: 
    	logits: the logits outputs of the model.
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_data_format() == 'channels_first':
        inputs = Input((3, 64, 64))
    else:
        inputs = Input((64, 64, 3))

    # Make inception base
    pred = ssrnet_base(inputs,s1,s2,s3,lambda_local,lambda_d)
   
    #print('pred:',pred)
    model = Model(inputs=inputs, outputs=pred, name='ssrnet')

    # load weights
    if weights == 'imdb':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path, by_name=True)
    return model


def create_model(s1,s2,s3,lambda_local,lambda_d, weights='imdb', include_top=True):
    return ssrnet(s1,s2,s3,lambda_local,lambda_d,weights, include_top)
