#import tensorflow as tf
import matplotlib.pyplot as plt
import sys

import models
import utils

model = models.get_model(sys.argv[1])
model.restore()
z = model.get_noise_sample(100)
gen_images = model.generate(z)
fig = utils.image_grid(gen_images, (10,10))

plt.show()
if len(sys.argv) == 3:
    fig.savefig(sys.argv[2], bbox_inches='tight')

