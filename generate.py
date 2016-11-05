import tensorflow as tf
import matplotlib.pyplot as plt

import models
import utils

with tf.Graph().as_default():
    model = models.MNIST_ALI()

model.restore()
z = model.get_noise_sample(100)
gen_images = model.generate(z)
utils.image_grid(gen_images, (10,10))

plt.show()

