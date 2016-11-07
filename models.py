import tensorflow as tf
from mnist import MNIST
from ali import ALI
from vae import VAE
from ae import AE
from vaegan import VAEGAN

def get_model(name):
    dataset, model = name.split('_')
    with tf.Graph().as_default():
        if dataset == 'MNIST':
            dataset = MNIST()
        else:
            raise NotImplementedError

        if model == 'ALI':
            return ALI(dataset)
        elif model == 'VAE':
            return VAE(dataset)
        elif model == 'AE':
            return AE(dataset)
        elif model == 'VAEGAN':
            return VAEGAN(dataset)
        else:
            raise NotImplementedError

