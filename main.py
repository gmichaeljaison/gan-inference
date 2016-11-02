import sys
import os.path as osp
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
from datasets import GanMnist, GanAliMnist
from gans.gan import GanBase, GanAli

_save_dir = 'models/'


def image_grid(images, size):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0.1)
    for i in range(size[0]*size[1]):
        im = np.reshape(images[i], (28,28))
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im, cmap='gray')
    return fig


def train(gan, steps, method):
    with tf.Graph().as_default():
        for step, gen_loss, disc_loss in gan.train(steps):
            if step % 100 == 0:
                print('Step {}, generator loss: {:.5f}, discriminator loss: {:.5f}'
                      .format(step, gen_loss, disc_loss))

    # TODO save model after training (tensorflow save?)
    fname = '{}-{}.ckpt'.format(method, int(time.time()))
    saver = tf.train.Saver()
    saver.save(gan.sess, osp.join(_save_dir, fname))
    print('model saved as {}'.format(fname))


def test(gan):
    gen_images = gan.generate(100)

    image_grid(gen_images, (10, 10))
    plt.show()


def main():
    method = sys.argv[1]
    steps = int(sys.argv[2])

    z_size = 64
    batch_size = 100

    gan = None
    if method == 'gan':
        dataset = GanMnist(z_size, batch_size)
        gan = GanBase(dataset)
    elif method == 'gan-ali':
        dataset = GanAliMnist(z_size, batch_size)
        gan = GanAli(dataset)

    train(gan, steps, method)
    test(gan)


if __name__ == '__main__':
    main()

