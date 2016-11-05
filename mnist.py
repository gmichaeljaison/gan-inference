import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.learn.python.learn.datasets import mnist

import custom_ops

INIT = lambda: tf.truncated_normal_initializer(stddev=0.01)

class MNIST:

    def __init__(self):
        data = mnist.read_data_sets('MNIST_data')
        # reshape
        data.train._images = data.train._images.reshape([-1, 28, 28, 1])
        data.validation._images = data.validation._images.reshape([-1, 28, 28, 1])
        data.test._images = data.test._images.reshape([-1, 28, 28, 1])
        self.data = data

        self.z_size = 32
        self.real_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.z_sampled = tf.placeholder(tf.float32, [None, self.z_size])

    def encoder(self, x):
        with tf.variable_scope('encoder'):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=custom_ops.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=INIT()):
                tmp = slim.conv2d(x, 32, 5, padding='VALID')
                tmp = slim.conv2d(tmp, 64, 4, stride=2, padding='VALID')
                tmp = slim.conv2d(tmp, 128, 4, padding='VALID')
                tmp = slim.conv2d(tmp, 256, 4, stride=2, padding='VALID')
                assert tmp.get_shape().as_list() == [None, 3, 3, 256]
                tmp = slim.flatten(tmp)
                tmp = slim.fully_connected(tmp, 512)
                tmp = slim.fully_connected(tmp, 512)
                mu = slim.fully_connected(tmp, self.z_size, activation_fn=None, normalizer_fn=None)
                log_sigma = slim.fully_connected(tmp, self.z_size, activation_fn=None, normalizer_fn=None)
                eps = tf.truncated_normal([tf.shape(mu)[0], self.z_size])
                z = mu + eps*tf.exp(log_sigma)
                #z = mu

                # FIXME: doing the reparameterization trick makes generator loss huge
                # it is not stricly necessary, but right now the encoder is deterministic
                return z

    def discriminator(self, x, reuse=False, ALI=False, get_features=False):
        if ALI:
            x, z = x
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=custom_ops.leaky_relu,
                                weights_initializer=INIT()):

                tmp = slim.dropout(x, keep_prob=0.8)
                tmp = slim.conv2d(tmp, 32, 5, padding='VALID')

                tmp = slim.dropout(tmp, keep_prob=0.5)
                tmp = slim.conv2d(tmp, 64, 4, stride=2, padding='VALID')

                tmp = slim.dropout(tmp, keep_prob=0.5)
                tmp = slim.conv2d(tmp, 128, 4, padding='VALID')
                
                tmp = slim.dropout(tmp, keep_prob=0.5)
                tmp = slim.conv2d(tmp, 256, 4, stride=2, padding='VALID')
                assert tmp.get_shape().as_list() == [None, 3, 3, 256]

                tmp = slim.flatten(tmp)

                # what layer should this be at?
                minibatch_discrim = custom_ops.minibatch_discrimination(tmp, 100)

                tmp = slim.dropout(tmp, keep_prob=0.5)
                tmp = slim.fully_connected(tmp, 512)


                if ALI:
                    tmp2 = slim.dropout(z, keep_prob=0.8)
                    tmp2 = slim.fully_connected(tmp2, 512)

                    tmp2 = slim.dropout(tmp2, keep_prob=0.5)
                    z_vector = slim.fully_connected(tmp2, 512)
                    tmp = tf.concat(1, [tmp, z_vector])

                tmp = slim.dropout(tmp, keep_prob=0.5)
                tmp = slim.fully_connected(tmp, 1024)

                tmp = slim.dropout(tmp, keep_prob=0.5)
                tmp = slim.fully_connected(tmp, 1024)

                tmp = slim.dropout(tmp, keep_prob=0.5)
                # what layer should this be concatenated at?
                tmp = tf.concat(1, [tmp, minibatch_discrim])
                logits = slim.fully_connected(tmp, 1, activation_fn=None)
                return logits

    def generator(self, z):
        with tf.variable_scope('generator'):
            with slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv2d_transpose],
                                activation_fn=custom_ops.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=INIT()):
                tmp = slim.fully_connected(z, 256*3*3)
                tmp = tf.reshape(tmp, [-1, 3, 3, 256])
                tmp = slim.conv2d_transpose(tmp, 128, 4, stride=2, padding='VALID')
                tmp = slim.conv2d_transpose(tmp, 64, 4, padding='VALID')
                tmp = slim.conv2d_transpose(tmp, 32, 4, stride=2, padding='VALID')
                tmp = slim.conv2d_transpose(tmp, 32, 5, padding='VALID')
                tmp = slim.conv2d(tmp, 32, 1, padding='VALID')
                gen_images = slim.conv2d(tmp, 1, 1, padding='VALID', activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                assert gen_images.get_shape().as_list() == [None, 28, 28, 1]
                return gen_images

    def next_batch(self, batch_size):
        return self.data.train.next_batch(batch_size)[0]

