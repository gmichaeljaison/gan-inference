import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import cv2 as cv

import models
import utils

from cv_utils import img_utils

model = models.get_model(sys.argv[1])
print('model', model.name)

model.restore()
print('1. restored')
z = model.get_noise_sample(100)
gen_images = model.generate(z)
# utils.image_grid(gen_images, (10,10))

imgs = list()
for i in range(gen_images.shape[0]):
    im = gen_images[i,:,:,:]
    im = img_utils.normalize_img(im)
    imgs.append(im)


coll = img_utils.collage(imgs, (10, 10))
print(coll.shape)
cv.imwrite('gen-{}.jpg'.format(model.name), coll)

# plt.show()

