import math
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
from cv_utils import img_utils


def image_grid(images, size, imsize):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0.1)
    for i in xrange(size[0]*size[1]):
        im = np.reshape(images[i], imsize)
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im, cmap='gray')
    return fig


def img_grid(im_tensor, size=None):
    imgs = list()
    n = im_tensor.shape[0]
    if size is None:
        nrows = int(math.sqrt(n))
        ncols = nrows
    else:
        nrows, ncols = size
    for i in range(n):
        im = im_tensor[i,:,:,:]
        im = img_utils.normalize_img(im)
        imgs.append(im)
    coll = img_utils.collage(imgs, (nrows, ncols))
    return coll

