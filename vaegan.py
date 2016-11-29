import tensorflow as tf
import numpy as np
import os.path

from model_base import Model_Base

class VAEGAN(Model_Base):

    def __init__(self, dataset):
        self.dataset = dataset
        self.encoder = self.dataset.encoder
        self.generator = self.dataset.generator
        self.discriminator = self.dataset.discriminator
        self.x_size = self.dataset.x_size
        self.z_size = self.dataset.z_size
        self.real_images = self.dataset.real_images
        self.z_sampled = self.dataset.z_sampled

        self.mu, self.log_sigma, self.encoder_features = self.encoder(self.real_images)
        eps = tf.truncated_normal([tf.shape(self.mu)[0], self.z_size])
        self.z_encoded = self.mu + eps*tf.exp(self.log_sigma)

        self.gen_images = self.generator(self.z_encoded)

        self.real_logits, self.real_features = self.discriminator(self.real_images, get_features=True)
        self.gen_logits, self.gen_features = self.discriminator(self.gen_images, get_features=True, reuse=True)

        self.generator_loss, self.discriminator_loss = self.losses()
        self.gen_train_op = self.get_train_op(self.generator_loss, net='generator')
        self.discrim_train_op = self.get_train_op(self.discriminator_loss, net='discriminator')
        
        Model_Base.__init__(self)
        self.init_op = tf.initialize_all_variables()

        self.name = self.dataset.name + '_VAEGAN'
        self.save_dir = os.path.join('./log', self.name)

    def losses(self):
        zeros = tf.zeros_like(self.real_logits)
        ones = tf.ones_like(self.real_logits)
        discrim_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.real_logits, ones))
        discrim_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_logits, zeros))
        discriminator_loss = discrim_loss1 + discrim_loss2

        output_sigma = 1.0
        fool_discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_logits, ones))
        reconstruction_loss = tf.reduce_mean(tf.square(self.real_features - self.gen_features))
        KL_loss = -output_sigma**2 * self.z_size/self.x_size*tf.reduce_mean(1 + 2*self.log_sigma - self.mu**2 - tf.exp(2*self.log_sigma))
        generator_loss = fool_discriminator_loss + reconstruction_loss + KL_loss

        return generator_loss, discriminator_loss

    def get_train_op(self, loss, net=None):
        if net == 'generator':
            variables = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                         + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder'))
        elif net == 'discriminator':
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        else:
            raise RuntimeError('Net to train must be one of: generator, discriminator')

        train_op = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(loss, var_list=variables)
        return train_op

    def train_one_step(self, image_batch):
        # train generator
        _, gen_loss_value = self.sess.run([self.gen_train_op, self.generator_loss],
                                          feed_dict={self.real_images: image_batch})
        assert not np.isnan(gen_loss_value), 'Model diverged with generator NaN loss value'

        # train discriminator
        _, discrim_loss_value = self.sess.run([self.discrim_train_op, self.discriminator_loss],
                                              feed_dict={self.real_images: image_batch})
        assert not np.isnan(discrim_loss_value), 'Model diverged with discriminator NaN loss value'

        return gen_loss_value, discrim_loss_value

    def train(self, steps):
        if tf.gfile.Exists(self.save_dir):
            tf.gfile.DeleteRecursively(self.save_dir)
        tf.gfile.MakeDirs(self.save_dir)

        self.sess.run(self.init_op)
        for i in range(steps):
            image_batch = self.dataset.next_batch()
            #z = self.get_noise_sample(image_batch.shape[0])
            gen_loss, discrim_loss = self.train_one_step(image_batch)

            if i % 10 == 0:
                print('Step {}, gen_loss = {}, discrim_loss = {}'.format(i, gen_loss, discrim_loss))

            if i % 1000 == 0 or (i+1) == steps:
                self.saver.save(self.sess, os.path.join(self.save_dir, self.name), global_step=i)

    def generate(self, z):
        gen_images = self.sess.run(self.gen_images, feed_dict={self.z_encoded: z})
        return gen_images

