import tensorflow as tf
import mnist
import cifar_dataset

class CIFAR(mnist.MNIST):
    def __init__(self):
        self.data = cifar_dataset.read_data_sets('CIFAR_data')

        self.z_size = 128
        self.x_size = 32*32
        self.ch = 3
        self.real_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.z_sampled = tf.placeholder(tf.float32, [None, self.z_size])

        self.s = 4

        self.name = 'CIFAR'

