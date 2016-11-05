import tensorflow as tf
import numpy as np

def leaky_relu(x, leakiness=0.01):
    assert leakiness <= 1
    return tf.maximum(x, leakiness * x)

def prelu(x):
    alphas = tf.get_variable('alpha', x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.))
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5
    return pos + neg

def minibatch_discrimination(input_layer, num_kernels, dim_per_kernel=5, name='minibatch_discrim'):
    shape = input_layer.get_shape().as_list()
    #batch_size = shape[0]
    num_features = shape[1]
    #shape = tf.shape(input_layer)
    #batch_size = shape[0]
    #num_features = shape[1]
    W = tf.get_variable('W', [num_features, num_kernels*dim_per_kernel],
                      initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', [num_kernels], initializer=tf.constant_initializer(0.0))
    activation = tf.matmul(input_layer, W)
    activation = tf.reshape(activation, [-1, num_kernels, dim_per_kernel])
    tmp1 = tf.expand_dims(activation, 3)
    tmp2 = tf.transpose(activation, perm=[1,2,0])
    tmp2 = tf.expand_dims(tmp2, 0)
    abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
    f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
    f = f + b
    return f

