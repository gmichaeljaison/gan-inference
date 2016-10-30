import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.learn.python.learn.datasets import mnist

class MNIST_base:
    
    def __init__(self, activation_fn=tf.nn.relu):
        self.data = mnist.read_data_sets('MNIST_data')
        self.activation_fn = activation_fn
        self.z_size = 64
        self.real_images = tf.placeholder(tf.float32, shape=[None, 784])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_size])
        #self.weights_initializer = tf.contrib.layers.xavier_initializer
        self.weights_initializer = lambda: tf.truncated_normal_initializer(stddev=0.01)

    def encoder(self, x):
        # not needed for GAN model
        # fill this in when developing other models
        pass

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d],
                                activation_fn=self.activation_fn,
                                weights_initializer=self.weights_initializer()):
                tmp = tf.reshape(x, [-1, 28, 28, 1])
                tmp = slim.conv2d(tmp, 128, 5, stride=2)
                tmp = slim.conv2d(tmp, 256, 5, stride=2)
                logits = slim.fully_connected(tmp, 1, activation_fn=None)
                return logits

    def generator(self, z):
        with tf.variable_scope('generator'):
            with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                                activation_fn=self.activation_fn,
                                weights_initializer=self.weights_initializer()):
                tmp = slim.fully_connected(z, 256*7*7)
                tmp = tf.reshape(tmp, [-1, 7, 7, 256])
                tmp = slim.conv2d_transpose(tmp, 128, 5, stride=2)
                tmp = slim.conv2d_transpose(tmp, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
                gen_images = tf.reshape(tmp, [-1, 784])
                return gen_images

    def next_batch(self, batch_size):
        return self.data.train.next_batch(batch_size)[0]

