import os
import os.path as osp
import numpy as np
import cv2 as cv
import tensorflow as tf
import tensorflow.contrib.slim as slim
import custom_ops

from glob import glob


INIT = lambda: tf.truncated_normal_initializer(stddev=0.01)


class CelebA:
    def __init__(self, batch_size):
        self.name = 'CelebA'
        self.batch_size = batch_size

        img_pattern = 'data/celeba-cropped/*.jpg'
        self.fpaths = glob(img_pattern)

        self.x_size = 64*64
        self.z_size = 256
        self.h, self.w, self.ch = 64, 64, 3
        self.real_images = tf.placeholder(tf.float32, shape=[None, self.h, self.w, self.ch])
        self.z_sampled = tf.placeholder(tf.float32, [None, self.z_size])

        # current batch index
        self.batch_i = 0

    def encoder(self, x):
        features = []
        with tf.variable_scope('encoder'):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=custom_ops.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=INIT()):
                tmp = slim.conv2d(x, 64, 2, padding='VALID')
                tmp = slim.conv2d(tmp, 128, 7, stride=2, padding='VALID')
                tmp = slim.conv2d(tmp, 256, 5, stride=2, padding='VALID')
                tmp = slim.conv2d(tmp, 256, 7, stride=2, padding='VALID')
                tmp = slim.conv2d(tmp, 512, 4, padding='VALID')
                assert tmp.get_shape().as_list() == [None, 1, 1, 512]
                tmp = slim.flatten(tmp)
                features.append(tmp)
                tmp = slim.fully_connected(tmp, 512)
                features.append(tmp)
                mu = slim.fully_connected(tmp, self.z_size, activation_fn=None, normalizer_fn=None)
                features.append(mu)
                log_sigma = slim.fully_connected(tmp, self.z_size, activation_fn=None, normalizer_fn=None)

                return mu, log_sigma, tf.concat(1, features)

    def discriminator(self, x, reuse=False, ALI=False, get_features=False):
        if ALI:
            x, z = x
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=custom_ops.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=INIT()):
                tmp = slim.conv2d(x, 64, 2, padding='VALID')
                tmp = slim.conv2d(tmp, 128, 7, stride=2, padding='VALID')
                tmp = slim.conv2d(tmp, 256, 5, stride=2, padding='VALID')
                tmp = slim.conv2d(tmp, 256, 7, stride=2, padding='VALID')
                tmp = slim.conv2d(tmp, 512, 4, padding='VALID')
                assert tmp.get_shape().as_list() == [None, 1, 1, 512]

                tmp = slim.flatten(tmp)

                # what layer should this be at?
                minibatch_discrim = custom_ops.minibatch_discrimination(tmp, 100)

                # what layer should this be at?
                features = tmp

                tmp = slim.dropout(tmp, keep_prob=0.8)
                tmp = slim.fully_connected(tmp, 1024)

                if ALI:
                    tmp2 = z
                    tmp2 = slim.fully_connected(tmp2, 1024, normalizer_fn=None)
                    tmp2 = slim.dropout(tmp2, keep_prob=0.8)

                    tmp2 = slim.fully_connected(tmp2, 1024, normalizer_fn=None)
                    tmp2 = slim.dropout(tmp2, keep_prob=0.8)
                    
                    tmp = tf.concat(1, [tmp, tmp2])

                tmp = slim.fully_connected(tmp, 2048, normalizer_fn=None)
                tmp = slim.dropout(tmp, keep_prob=0.8)
                
                tmp = slim.fully_connected(tmp, 2048, normalizer_fn=None)
                tmp = slim.dropout(tmp, keep_prob=0.8)

                # what layer should this be concatenated at?
                tmp = tf.concat(1, [tmp, minibatch_discrim])
                
                logits = slim.fully_connected(tmp, 1, normalizer_fn=None, activation_fn=None)
                if get_features:
                    return logits, features
                return logits

    def generator(self, z):
        with tf.variable_scope('generator'):
            with slim.arg_scope([slim.fully_connected, slim.conv2d, slim.conv2d_transpose],
                                activation_fn=custom_ops.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=INIT()):
                # tmp = z
                tmp = slim.fully_connected(z, 512)
                tmp = tf.reshape(tmp, [-1, 1, 1, 512])
                tmp = slim.conv2d_transpose(tmp, 512, 4, padding='VALID')
                tmp = slim.conv2d_transpose(tmp, 256, 7, stride=2, padding='VALID')
                tmp = slim.conv2d_transpose(tmp, 256, 5, stride=2, padding='VALID')
                tmp = slim.conv2d_transpose(tmp, 128, 7, stride=2, padding='VALID')
                tmp = slim.conv2d_transpose(tmp, 64, 2, padding='VALID')
                gen_images = slim.conv2d(tmp, 3, 1, padding='VALID', activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                assert gen_images.get_shape().as_list() == [None, 64, 64, 3]
                return gen_images

    def next_batch(self):
        batch = np.zeros((self.batch_size, self.h, self.w, self.ch))

        i, n = self.batch_i, 0
        while n < self.batch_size:  
            im = cv.imread(self.fpaths[i])
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB) # add this line
            im = im.astype(np.float32) / 255. # add this line

            batch[n, :, :, :] = im

            i, n = i+1, n+1

            if i == len(self.fpaths):
                self.batch_i, i = 0, 0
        self.batch_i = i
        return batch


def crop_and_resize(im, crop_s, reshape=None):
    h, w = im.shape[:2]
    j = int(round((h - crop_s[0]) / 2.0))
    i = int(round((w - crop_s[1]) / 2.0))
    crop_im = im[j:j+crop_s[0], i:i+crop_s[1], :]
    if reshape is not None:
        crop_im = cv.resize(crop_im, reshape)
    return crop_im


def crop_and_save():
    from cv_utils import img_utils

    data_dir = 'data/celeba'
    out_dir = 'data/celeba-cropped'
    for n, fname in enumerate(os.listdir(data_dir)):
        print(n)
        im = cv.imread(osp.join(data_dir, fname))
        cim = crop_and_resize(im, (120, 120), (64, 64))
        cv.imwrite(osp.join(out_dir, 'crop-{}.jpg'.format(n+1)), cim)


if __name__ == '__main__':
    crop_and_save()

