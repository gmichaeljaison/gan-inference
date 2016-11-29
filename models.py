import tensorflow as tf

from celeba import CelebA
from mnist import MNIST
from ali import ALI
from vae import VAE
from ae import AE
from vaegan import VAEGAN


_batch_size = 64


def get_model(name):
    dataset, model = name.split('_')
    with tf.Graph().as_default():
        if dataset == 'MNIST':
            dataset = MNIST(_batch_size)
        elif dataset == 'CelebA':
            dataset = CelebA(_batch_size)
        else:
            raise NotImplementedError

        if model == 'ALI':
            print('michael i sgood')
            return ALI(dataset)
        elif model == 'VAE':
            return VAE(dataset)
        elif model == 'AE':
            return AE(dataset)
        elif model == 'VAEGAN':
            return VAEGAN(dataset)
        else:
            raise NotImplementedError

