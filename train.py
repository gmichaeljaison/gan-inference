import sys
import cv2 as cv
import models
import utils
import matplotlib.pyplot as plt
from cv_utils import img_utils

model = models.get_model(sys.argv[1])
model.train(int(sys.argv[2]))
print('train end')

z = model.get_noise_sample(100)
gen_images = model.generate(z)
print(gen_images.shape)
print(gen_images[0,:,:,:].max())

# imgs = list()
# for i in range(gen_images.shape[0]):
#     im = gen_images[i,:,:,:]
#     im = img_utils.normalize_img(im)
#     imgs.append(im)

utils.image_grid(gen_images, (10,10), (28,28))
plt.show()

# coll = img_utils.collage(imgs, (10, 10))
# print(coll.shape)
# cv.imwrite('gen.jpg', coll)



