import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import cv2 as cv
import numpy as np
import models
import utils

from cv_utils import img_utils


model = models.get_model(sys.argv[1])
print('model', model.name)

model.restore()
print('1. restored')

"""
z = model.get_noise_sample(100)
gen_images = model.generate(z)
# utils.image_grid(gen_images, (10,10))


imgs = list()
for i in range(gen_images.shape[0]):
    im = gen_images[i,:,:,:]
    im = img_utils.normalize_img(im)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    imgs.append(im)


coll = img_utils.collage(imgs, (10, 10))
print(coll.shape)
cv.imwrite('gen-{}.jpg'.format(model.name), coll)

# plt.show()
"""

image_batch = model.dataset.next_batch()
z = model.inference(image_batch)
gen_images = model.generate(z)

imgs = list()
for i in range(image_batch.shape[0]):
    im = image_batch[i,:,:,:]
    im = img_utils.normalize_img(im)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    imgs.append(im)


coll = img_utils.collage(imgs, (10, 10))
cv.imwrite('real-{}.jpg'.format(model.name), coll)



imgs1 = list()
for i in range(gen_images.shape[0]):
    im = gen_images[i,:,:,:]
    im = img_utils.normalize_img(im)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    imgs1.append(im)

coll1 = img_utils.collage(imgs1, (10, 10))
cv.imwrite('gen-{}.jpg'.format(model.name), coll1)

image_batch = model.dataset.next_batch()
s = cv.imread('1.jpg')
s = cv.cvtColor(s, cv.COLOR_RGB2BGR)
s = cv.resize(s, (64,64))
image_batch[9,:,:,:] = s
z1 = model.inference(image_batch[0:10,:,:,:])
image_batch = model.dataset.next_batch()
s = cv.imread('2.jpg')
s = cv.resize(s, (64,64))
s = cv.cvtColor(s, cv.COLOR_RGB2BGR)
image_batch[9,:,:,:] = s
z2 = model.inference(image_batch[0:10,:,:,:])
z1 = np.array(z1)
z2 = np.array(z2)

imgs2 = list()
g1 = model.generate(z1)
g2 = model.generate(z2)
for i1 in range(g1.shape[0]):
    im = g1[i1, :, :, :]
    im = img_utils.normalize_img(im)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    imgs2.append(im)
for i1 in range(g2.shape[0]):
    im = g2[i1, :, :, :]
    im = img_utils.normalize_img(im)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    imgs2.append(im)

coll2 = img_utils.collage(imgs2, (10, 10))
cv.imwrite('gen1-2-{}.jpg'.format(model.name), coll2)



imgs3 = list()

for i in range(11):
    z_new = (z1*(10.0-i)/10.0) + (z2*(i*1.0)/10.0)
    gen_images = model.generate(z_new)
    temp = []
    for i1 in range(gen_images.shape[0]):
        im = gen_images[i1, :, :, :]
        im = img_utils.normalize_img(im)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        imgs3.append(im)

print('----')
coll3 = img_utils.collage(imgs3, (11, 10))
cv.imwrite('interpolated-{}.jpg'.format(model.name), coll3)
print('----')

