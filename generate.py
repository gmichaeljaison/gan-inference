#import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import cv2 as cv

import models
import utils

from cv_utils import img_utils

print('hello')
print(sys.argv[1])
model = models.get_model(sys.argv[1])
print('model', model.name)
model.restore()
print('1. restored')
z = model.get_noise_sample(100)
gen_images = model.generate(z)
# utils.image_grid(gen_images, (10,10))

coll = img_utils.collage(gen_images, (10, 10))
print(coll.shape)
cv.imwrite('gen.jpg', coll)

# plt.show()

