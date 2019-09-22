import tensorflow as tf

def weight_variable(shape, stddev=0.02, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv3d(x,W,b):
    with tf.name_scope("conv3d"):
        conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv_3d_b = tf.nn.bias_add(conv_3d, b)
        return conv_3d_b

def Unidirectional_pool(x,n):
    return tf.nn.max_pool3d(x, ksize=[1, n, 1, 1, 1], strides=[1, n, 1, 1, 1], padding='SAME')
