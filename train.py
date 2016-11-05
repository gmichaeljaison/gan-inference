import tensorflow as tf
import sys

import models

with tf.Graph().as_default():
    model = models.MNIST_ALI()

model.train(int(sys.argv[1]), 64)

