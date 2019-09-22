import numpy as np
import tensorflow as tf

def cross_entropy(y_,y,class_weights=None):
    if class_weights is not None:
        flat_logits = tf.reshape(y_, [-1, 2])
        flat_labels = tf.reshape(y, [-1, 1])
        loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits, labels=tf.squeeze(flat_labels,squeeze_dims=[-1]),name="entropy")
        #loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
        #                                                      labels=flat_labels)
        flat_labels = tf.cast(flat_labels,'float32')
        class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

        weight_map = tf.multiply(flat_labels, class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)


        weighted_loss = tf.multiply(loss_map, weight_map)

        loss = tf.reduce_mean(weighted_loss)
        return loss
    else:
        return tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.squeeze(y,squeeze_dims=[-1]),name="entropy")))


