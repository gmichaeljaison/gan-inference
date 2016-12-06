import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def image_grid(images, size):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=size, axes_pad=0.1)
    side = int(images.shape[1])
    if images.ndim == 3:
        ch = 1
    else:
        ch = 3
    for i in xrange(size[0]*size[1]):
        im = np.reshape(images[i], (side,side,ch))
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im, cmap='gray')
    return fig

