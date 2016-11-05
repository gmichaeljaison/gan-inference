import tensorflow as tf
import matplotlib.pyplot as plt
import models
import utils

with tf.Graph().as_default():
    model = models.MNIST_ALI()
model.restore()

image_batch = model.data.validation.next_batch(100)[0]
z = model.inference(image_batch)
gen_images = model.generate(z)

utils.image_grid(image_batch, (10,10))
utils.image_grid(gen_images, (10,10))

plt.show()

