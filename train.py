import sys
import cv2 as cv
import models
import logging

logging.basicConfig(level=logging.DEBUG)


model = models.get_model(sys.argv[1])
logging.info('Model created: {}'.format(model.name))

model.train(int(sys.argv[2]))



