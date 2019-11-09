import tensorflow as tf
import numpy as np
import utils
from layers import (weight_variable,bias_variable,conv3d,Unidirectional_pool)

def IPN(x,PLM_NUM=5,filter_size=[3,3,3],LAYER_NUM=3,NUM_OF_CLASS=2,pooling_size=[]):
    """
    :param x: input tensor,shape[?,nx,ny,nz,channels]
    :param filter_size: size of the convolution filer
    :param PLM_NUM: number of PLM
    :param LAYER_NUM: number of conv layers in each PLM
    :return:
    """
    #Initialize Variable
    #whether use pooling size by automatic or by yourself
    if not pooling_size:
        pooling_size = utils.cal_downsampling_size_combine(x.shape[1],PLM_NUM)
    else:
        PLM_NUM = len(pooling_size)

    W=[[0]*LAYER_NUM for i in range(PLM_NUM)]
    b=[[0]*LAYER_NUM for i in range(PLM_NUM)]
    conv =[[0]*LAYER_NUM for i in range(PLM_NUM)]
    pool =[[0]*LAYER_NUM for i in range(PLM_NUM)]
    variables = []

    #features = utils.cal_channel_num(PLM_NUM)
    features = np.ones(PLM_NUM,dtype='int32')*64
    ##################### print model para  #############
    print('')
    print('-----------------  model paras ------------------')
    resize = 1
    print('PLM DS SIZE: ',end='')
    for index in pooling_size:
        resize *= index
    print('{}->{} = '.format(x.shape[1],resize),end='')
    for i,s in enumerate(pooling_size):
        if i == 0:
            print(str(s),end='')
        else:
            print('x'+str(s),end='')

    print('')
    print('conv channel nums : ',end='')
    for f in features:
        print(f,',',end='')
    print('')
    print('---------------------  end ----------------------')
    print('')
    ######################################################

    features_count = -1
    stddev = 0.02

    #Build Projection Learning Module
    for PLM in range(PLM_NUM):
        features_count += 1
        if PLM == 0:
            input=x
        else:
            input=pool[PLM-1]
        for LAYER in range(LAYER_NUM):
            b[PLM][LAYER] = bias_variable([features[features_count]], name="b{}".format(PLM+1))
            in_channels = input.get_shape().as_list()[-1]
            W[PLM][LAYER] = weight_variable(filter_size + [in_channels, features[features_count]], stddev, name="w{}_{}".format(PLM+1, LAYER+1))
            variables.append(W[PLM][LAYER])
            variables.append(b[PLM][LAYER])
            conv[PLM][LAYER] = tf.nn.relu(conv3d(input, W[PLM][LAYER], b[PLM][LAYER]))
            if LAYER == LAYER_NUM-1:
                pool[PLM]=Unidirectional_pool(conv[PLM][LAYER], pooling_size[PLM])#Unidirectional_pool
            else:
                input = conv[PLM][LAYER]

    #Output MAP
    Wop = weight_variable(filter_size + [features[features_count], NUM_OF_CLASS], stddev, name="w_output")
    bop = bias_variable([NUM_OF_CLASS], name="b_output")
    output= tf.nn.relu(tf.nn.bias_add(tf.nn.conv3d(pool[PLM_NUM-1], Wop, strides=[1, 1, 1, 1, 1], padding="SAME"), bop))

    sf = tf.nn.softmax(output)
    pred = tf.argmax(sf, axis=-1, name="prediction")
    return output,pred,variables,sf