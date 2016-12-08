import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def image_grid(images, size):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0.1)
    side = int(images.shape[1])
    ch = images.shape[3]
    if ch == 3:
        cmap = None
    else:
        cmap = 'gray'
    for i in xrange(size[0]*size[1]):
        #im = np.reshape(images[i], (side,side,ch))
        im = np.squeeze(images[i])
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im, cmap=cmap)
    return fig

def images_and_reconstructions(images, reconstructions, size):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0.1)
    ch = images.shape[3]
    if ch == 3:
        cmap = None
    else:
        cmap = 'gray'
    for i in xrange(size[0]*size[1]/2):
        im = np.squeeze(images[i])
        rec = np.squeeze(reconstructions[i])
        axis1 = grid[2*i]
        axis2 = grid[2*i+1]
        axis1.axis('off')
        axis2.axis('off')
        axis1.imshow(im, cmap=cmap)
        axis2.imshow(rec, cmap=cmap)
    return fig

