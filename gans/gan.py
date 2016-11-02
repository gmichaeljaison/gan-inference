import collections
import numpy as np
import tensorflow as tf


Network = collections.namedtuple('Network', ['input', 'output'])
Discriminator = collections.namedtuple('Discriminator',
                                       ['real_inp', 'gen_inp',
                                        'real_logits', 'gen_logits'])


class GanBase:
    def __init__(self, dataset):
        self.dataset = dataset
        self.optimizer = tf.train.AdamOptimizer(0.0003, beta1=0.5)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator_loss, self.discriminator_loss = self.losses()

        self.gen_train_op = self.generator_train_op()
        self.disc_train_op = self.discriminator_train_op()

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def build_generator(self):
        inp = self.dataset.z
        out = self.dataset.generator(inp)
        return Network(input=inp, output=out)

    def build_discriminator(self):
        real_inp = self.get_disc_real_input()
        gen_inp = self.get_disc_gen_input()

        real_logits = self.dataset.discriminator(self.get_disc_real_input())
        gen_logits = self.dataset.discriminator(self.get_disc_gen_input(), reuse=True)

        discriminator = Discriminator(real_inp, gen_inp, real_logits, gen_logits)
        return discriminator

    def get_disc_real_input(self):
        return self.dataset.x

    def get_disc_gen_input(self):
        return self.generator.output

    def losses(self):
        real_logits, gen_logits = self.discriminator.real_logits, self.discriminator.gen_logits
        ones, zeros = tf.ones_like(real_logits), tf.zeros_like(real_logits)

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_logits, ones))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_logits, zeros))
        discriminator_loss = tf.add(real_loss, fake_loss, name='discriminator_loss')

        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gen_logits, ones),
                                        name='generator_loss')

        return generator_loss, discriminator_loss

    def generator_train_op(self):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        return self.optimizer.minimize(self.generator_loss, var_list=variables)

    def discriminator_train_op(self):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return self.optimizer.minimize(self.discriminator_loss, var_list=variables)

    def get_z_sample(self, batch_size):
        return np.random.uniform(low=-1, high=1, size=(batch_size, self.dataset.z_size))

    def train_one_step(self, x_batch, z):
        # train generator
        # FIXME is feed_dict common? does it work?
        _, gen_loss_value = self.sess.run(
            [self.gen_train_op, self.generator_loss],
            feed_dict={self.dataset.x: x_batch, self.dataset.z: z})
        assert not np.isnan(gen_loss_value), 'Model diverged with generator NaN loss value'

        # train discriminator
        _, disc_loss_value = self.sess.run(
            [self.disc_train_op, self.discriminator_loss],
            feed_dict={self.dataset.x: x_batch, self.dataset.z: z})
        assert not np.isnan(disc_loss_value), 'Model diverged with discriminator NaN loss value'

        return gen_loss_value, disc_loss_value

    def train(self, steps, batch_size=None):
        for i in range(steps):
            x_batch = self.dataset.next_batch(batch_size)
            z = self.get_z_sample(x_batch.shape[0])
            gen_loss, disc_loss = self.train_one_step(x_batch, z)

            yield i, gen_loss, disc_loss

    def generate(self, num_images):
        z = self.get_z_sample(num_images)
        gen_images = self.sess.run(self.generator.output, feed_dict={self.dataset.z: z})
        return gen_images


class GanAli(GanBase):
    def __init__(self, dataset):
        GanBase.__init__(self, dataset)

    def build_generator(self):
        eps = tf.truncated_normal([self.dataset.batch_size, self.dataset.z_size])
        mu, log_sigma = self.dataset.encoder(self.dataset.x)
        encoding = mu + eps * tf.exp(log_sigma)

        gen_images = self.dataset.generator(self.dataset.z)

        inp = (self.dataset.x, self.dataset.z)
        out = (encoding, gen_images)
        return Network(input=inp, output=out)

    def get_disc_real_input(self):
        # corresponds to the encoder network
        return self.generator.input[0], self.generator.output[0]

    def get_disc_gen_input(self):
        # corresponds to the decoder network
        return self.generator.output[1], self.generator.input[1]

    def losses(self):
        gen_loss, disc_loss = GanBase.losses(self)

        real_logits = self.discriminator.real_logits
        zeros = tf.zeros_like(real_logits)
        encoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_logits, zeros),
                                      name='encoder_loss')

        gen_loss = tf.add(gen_loss, encoder_loss, name='generator_loss')
        return gen_loss, disc_loss

    def generator_train_op(self):
        gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        enc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')
        return self.optimizer.minimize(self.generator_loss, var_list=gen_variables + enc_variables)

    def generate(self, num_images):
        z = self.get_z_sample(num_images)
        gen_images = self.sess.run(self.generator.output[1], feed_dict={self.dataset.z: z})
        return gen_images
