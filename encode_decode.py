#import tensorflow as tf
import matplotlib.pyplot as plt
import sys

import models
import utils

model = models.get_model(sys.argv[1])
model.restore()

image_batch = model.dataset.data.validation.next_batch(100)[0]
z = model.inference(image_batch)
gen_images = model.generate(z)

utils.image_grid(image_batch, (10,10))
fig = utils.image_grid(gen_images, (10,10))

plt.show()
if len(sys.argv) == 3:
    fig.savefig(sys.argv[2], bbox_inches='tight')

