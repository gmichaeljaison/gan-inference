import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sys

import mnist
import gan

class GAN_MNIST(mnist.MNIST_base, gan.GAN_base):
    def __init__(self, activation_fn=tf.nn.relu):
        mnist.MNIST_base.__init__(self, activation_fn=activation_fn)
        gan.GAN_base.__init__(self)

def image_grid(images, size):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0.1)
    for i in xrange(size[0]*size[1]):
        im = np.reshape(images[i], (28,28))
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im, cmap='gray')
    return fig


with tf.Graph().as_default():
    model = GAN_MNIST()

model.train(int(sys.argv[1]), 64)
gen_images = model.generate(100)
image_grid(gen_images, (10,10))

plt.show()

