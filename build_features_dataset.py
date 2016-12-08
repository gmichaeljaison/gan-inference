#import tensorflow as tf
import numpy as np
import sys
import pickle

import models

model = models.get_model(sys.argv[1])
model.restore()
data = model.dataset.data

features_dataset = {}
for name, dataset in [('train', data.train),
                      ('validation', data.validation),
                      ('test', data.test)]:
    N = dataset._images.shape[0]
    features_dim = model.encoder_features.get_shape().as_list()[1]
    features = np.zeros((N, features_dim), dtype=np.float32)
    for i in xrange(N):
        im = dataset._images[i:i+1]
        features[i] = model.get_encoder_features(im)
    features_dataset[name] = (features, dataset._labels)

with open('./features_datasets/%s.pkl' % model.name, 'wb') as f:
    pickle.dump(features_dataset, f, 2)

'''
for dataset in [data.train, data.validation, data.test]:
    N = dataset._images.shape[0]
    zs = np.zeros((N, model.z_size))
    for i in xrange(N):
        im = dataset._images[i:i+1]
        zs[i] = model.inference(im)
    dataset._images = zs

with open('./features_datasets/MNIST_ALI.pkl', 'w') as f:
    pickle.dump(data, f)
'''

