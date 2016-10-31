import numpy as np
import tensorflow as tf

from gan import GAN_base


class GAN_ALI(GAN_base):
    def __init__(self, dataset):
        # super().__init__()
        self.dataset = dataset
        self.encoding = dataset.encoder(dataset.real_images)
        self.gen_images = dataset.generator(dataset.z)

        self.real_logits = dataset.discriminator((dataset.real_images, self.encoding))
        self.gen_logits = dataset.discriminator((self.gen_images, dataset.z), reuse=True)

        self.encoder_loss, self.gen_loss, self.disc_loss = self.losses()
        self.gen_train_op = self.get_train_op(self.gen_loss, net='generator')
        self.discrim_train_op = self.get_train_op(self.disc_loss, net='discriminator')
        self.encoder_train_op = self.get_train_op(self.encoder_loss, net='encoder')

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def losses(self):
        gen_loss, disc_loss = super().losses()
        encoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.real_logits, tf.ones_like(self.real_logits)), name='encoder_loss')
        return encoder_loss, gen_loss, disc_loss

    def train_one_step(self, image_batch, z):
        # encoder
        _, enc_loss_val = self.sess.run([self.encoder_train_op, self.encoder_loss],
                      feed_dict={self.dataset.real_images: image_batch})
        assert not np.isnan(enc_loss_val), 'Model diverged with encoder NaN loss value'

        gen_loss_val, discrim_loss_val = super().train_one_step(image_batch, z)
        return enc_loss_val + gen_loss_val, discrim_loss_val
