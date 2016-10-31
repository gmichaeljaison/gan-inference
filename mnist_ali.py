import tensorflow as tf
import tensorflow.contrib.slim as slim

from mnist import MNIST_base


class MNIST_ALI(MNIST_base):
    def __init__(self):
        super().__init__()

    def encoder(self, x):
        with tf.variable_scope('encoder'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=self.activation_fn,
                                weights_initializer=self.weights_initializer()):
                tmp = tf.reshape(x, [-1, 28, 28, 1])
                tmp = slim.conv2d(tmp, 128, 5, stride=2)
                tmp = slim.conv2d(tmp, 256, 5, stride=2)
                enc = slim.fully_connected(tmp, self.z_size, activation_fn=None)
                return enc

    def discriminator(self, x, reuse=False):
        x, z = x
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=self.activation_fn,
                                weights_initializer=self.weights_initializer()):
                tmp = tf.reshape(x, [-1, 28, 28, 1])
                tmp = slim.conv2d(tmp, 128, 5, stride=2)
                tmp = slim.conv2d(tmp, 256, 5, stride=2)
                x_disc = slim.fully_connected(tmp, 128, activation_fn=None)

                tmp = slim.fully_connected(z, 128)
                z_disc = slim.fully_connected(tmp, 128, activation_fn=None)

                joint_inp = tf.concat(1, (x_disc, z_disc))
                tmp = slim.fully_connected(joint_inp, 256)
                tmp = slim.fully_connected(tmp, 256)
                logits = slim.fully_connected(tmp, 1, activation_fn=None)
                return logits
