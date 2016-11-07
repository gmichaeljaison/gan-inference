import tensorflow as tf
import os.path

from vae import VAE

class AE(VAE):

    def __init__(self, dataset):
        VAE.__init__(self, dataset)
        self.loss = self.reconstruction_loss
        self.train_op = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.loss)
        self.init_op = tf.initialize_all_variables()

        self.name = self.dataset.name + '_AE'
        self.save_dir = os.path.join('./log', self.name)

