import numpy as np
import tensorflow as tf

from gan import GAN_base


class GAN_ALI(GAN_base):
    def __init__(self, dataset):
        # super().__init__()
        self.dataset = dataset

        # self.encoding = dataset.encoder(dataset.real_images)
        eps = tf.truncated_normal([None, self.dataset.z_size])
        mu, log_sigma_sq = dataset.encoder(dataset.real_images)
        self.encoding = mu + eps * tf.sqrt(tf.exp(log_sigma_sq))

        self.gen_images = dataset.generator(dataset.z)

        self.real_logits = dataset.discriminator((dataset.real_images, self.encoding))
        self.gen_logits = dataset.discriminator((self.gen_images, dataset.z), reuse=True)

        # self.encoder_loss, self.generator_loss, self.discriminator_loss = self.losses()
        self.generator_loss, self.discriminator_loss = self.losses()
        self.gen_train_op = self.generator_train_op()
        self.discrim_train_op = self.discriminator_train_op()
        # self.gen_train_op = self.get_train_op(self.generator_loss, net='generator')
        # self.discrim_train_op = self.get_train_op(self.discriminator_loss, net='discriminator')
        # self.encoder_train_op = self.get_train_op(self.encoder_loss, net='encoder')

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def losses(self):
        gen_loss, disc_loss = GAN_base.losses(self)
        encoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.real_logits, tf.zeros_like(self.real_logits)), name='encoder_loss')
        gen_loss = tf.add(gen_loss, encoder_loss, name='generator_loss')
        # return encoder_loss, gen_loss, disc_loss
        return gen_loss, disc_loss

    def generator_train_op(self):
        gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        enc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator1')
        return GAN_base.get_train_op(self.generator_loss, gen_variables + enc_variables)

    def train_one_step(self, image_batch, z):
        # encoder
        # _, enc_loss_val = self.sess.run([self.encoder_train_op, self.encoder_loss],
        #               feed_dict={self.dataset.real_images: image_batch})
        # assert not np.isnan(enc_loss_val), 'Model diverged with encoder NaN loss value'

        gen_loss_val, discrim_loss_val = GAN_base.train_one_step(self, image_batch, z)
        return gen_loss_val, discrim_loss_val
