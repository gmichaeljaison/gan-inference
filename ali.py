import tensorflow as tf
import numpy as np
import os.path

from model_base import Model_Base

class ALI(Model_Base):

    def __init__(self):
        self.gen_images = self.generator(self.z_sampled)

        mu, log_sigma, self.encoder_features = self.encoder(self.real_images)
        eps = tf.truncated_normal([tf.shape(mu)[0], self.z_size])
        self.z_encoded = mu + eps*tf.exp(log_sigma)

        self.real_logits = self.discriminator((self.real_images, self.z_encoded), ALI=True)
        self.gen_logits = self.discriminator((self.gen_images, self.z_sampled), ALI=True, reuse=True)

        self.generator_loss, self.discriminator_loss = self.losses()
        self.gen_train_op = self.get_train_op(self.generator_loss, net='generator')
        self.discrim_train_op = self.get_train_op(self.discriminator_loss, net='discriminator')
        
        Model_Base.__init__(self)
        self.init_op = tf.initialize_all_variables()

    def losses(self):
        zeros = tf.zeros_like(self.real_logits)
        ones = tf.ones_like(self.real_logits)
        discrim_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.real_logits, ones))
        discrim_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_logits, zeros))
        discriminator_loss = tf.add(discrim_loss1, discrim_loss2, name='discriminator_loss')

        gen_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_logits, ones))
        gen_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.real_logits, zeros))
        generator_loss = tf.add(gen_loss1, gen_loss2, name='generator_loss')

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

    def train_one_step(self, image_batch, z):
        # train generator
        _, gen_loss_value = self.sess.run([self.gen_train_op, self.generator_loss],
                                          feed_dict={self.real_images: image_batch,
                                                     self.z_sampled: z})
        assert not np.isnan(gen_loss_value), 'Model diverged with generator NaN loss value'

        # train discriminator
        _, discrim_loss_value = self.sess.run([self.discrim_train_op, self.discriminator_loss],
                                              feed_dict={self.real_images: image_batch,
                                                         self.z_sampled: z})
        assert not np.isnan(discrim_loss_value), 'Model diverged with discriminator NaN loss value'

        return gen_loss_value, discrim_loss_value

    def train(self, steps, batch_size):
        if tf.gfile.Exists(self.save_dir):
            tf.gfile.DeleteRecursively(self.save_dir)
        tf.gfile.MakeDirs(self.save_dir)

        self.sess.run(self.init_op)
        for i in xrange(steps):
            image_batch = self.next_batch(batch_size)
            z = self.get_noise_sample(image_batch.shape[0])
            gen_loss, discrim_loss = self.train_one_step(image_batch, z)

            if i % 10 == 0:
                print 'Step %d, gen_loss = %f, discrim_loss = %f' % (i, gen_loss, discrim_loss)

            if i % 1000 == 0 or (i+1) == steps:
                self.saver.save(self.sess, os.path.join(self.save_dir, self.name), global_step=i)

