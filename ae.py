import tensorflow as tf

from vae import VAE

class AE(VAE):

    def __init__(self):
        VAE.__init__(self)
        self.loss = self.reconstruction_loss
        self.train_op = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.loss)
        self.init_op = tf.initialize_all_variables()

