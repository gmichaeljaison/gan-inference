import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.learn.python.learn.datasets import mnist
from dataset.dataset import GanDataset


class GanMnist(GanDataset):
    """
    GAN Architecture for MNIST dataset
    """
    def __init__(self, z_size, batch_size=100):
        GanDataset.__init__(self, z_size, batch_size)

    def read_data(self):
        return mnist.load_mnist()

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d],
                                activation_fn=self.activation,
                                weights_initializer=self.weights_initializer()):
                tmp = tf.reshape(x, [-1, 28, 28, 1])
                tmp = slim.conv2d(tmp, 128, 5, stride=2)
                tmp = slim.conv2d(tmp, 256, 5, stride=2)
                logits = slim.fully_connected(tmp, 1, activation_fn=None)
                return logits

    def generator(self, z):
        with tf.variable_scope('generator'):
            with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                                activation_fn=self.activation,
                                weights_initializer=self.weights_initializer()):
                tmp = slim.fully_connected(z, 256*7*7)
                tmp = tf.reshape(tmp, [-1, 7, 7, 256])
                tmp = slim.conv2d_transpose(tmp, 128, 5, stride=2)
                tmp = slim.conv2d_transpose(tmp, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
                gen_images = tf.reshape(tmp, [-1, self.x_size])
                return gen_images


class GanAliMnist(GanMnist):
    """
    GAN-ALI architecture for MNIST dataset
    """
    def __init__(self, z_size, batch_size=100):
        GanMnist.__init__(self, z_size, batch_size)

    def encoder(self, x):
        with tf.variable_scope('encoder'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=self.activation,
                                weights_initializer=self.weights_initializer()):
                tmp = tf.reshape(x, [-1, 28, 28, 1])
                tmp = slim.conv2d(tmp, 128, 5, stride=2)
                tmp = slim.conv2d(tmp, 256, 5, stride=2)
                tmp = tf.reshape(tmp, [-1, 7*7*256])
                mu = slim.fully_connected(tmp, self.z_size, activation_fn=None)
                log_sigma = slim.fully_connected(tmp, self.z_size, activation_fn=None)
                return mu, log_sigma

    def discriminator(self, x, reuse=False):
        """
        Discriminator for GAN-ALI that tries to discriminate between real (x,z_cap) distribution
            from generated (x_cap, z) distribution

        :param x: (x, z) Both input image and its encoding
        :param reuse: If true, the same network parameters will be reused.
            Required for real and generated inputs
        :return: Output of the discriminator network
        """
        x, z = x
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=self.activation,
                                weights_initializer=self.weights_initializer()):
                tmp = tf.reshape(x, [-1, 28, 28, 1])
                tmp = slim.conv2d(tmp, 128, 5, stride=2)
                tmp = slim.conv2d(tmp, 256, 5, stride=2)
                tmp = tf.reshape(tmp, [-1, 7*7*256])
                x_disc = slim.fully_connected(tmp, 128)

                tmp = slim.fully_connected(z, 128)
                z_disc = slim.fully_connected(tmp, 128)

                joint_inp = tf.concat(1, [x_disc, z_disc])
                tmp = slim.fully_connected(joint_inp, 256)
                tmp = slim.fully_connected(tmp, 256)
                logits = slim.fully_connected(tmp, 1, activation_fn=None)
                return logits
