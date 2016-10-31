import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class GAN_base:

    def __init__(self):
        self.gen_images = self.generator(self.z)

        self.real_logits = self.discriminator(self.real_images)
        self.gen_logits = self.discriminator(self.gen_images, reuse=True)

        self.generator_loss, self.discriminator_loss = self.losses()
        self.gen_train_op = self.get_train_op(self.generator_loss, net='generator')
        self.discrim_train_op = self.get_train_op(self.discriminator_loss, net='discriminator')

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


    def losses(self):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.real_logits, tf.ones_like(self.real_logits)))
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_logits, tf.zeros_like(self.gen_logits)))
        discriminator_loss = tf.add(real_loss, gen_loss, name='discriminator_loss')

        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.gen_logits, tf.ones_like(self.gen_logits)), name='generator_loss')
        return generator_loss, discriminator_loss

    def get_train_op(self, loss, net=None):
        if net == 'generator':
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        elif net == 'discriminator':
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        else:
            raise RuntimeError('Net to train must be one of: generator, discriminator')

        train_op = tf.train.AdamOptimizer(0.0003, beta1=0.5).minimize(loss, var_list=variables)
        #train_op = tf.train.MomentumOptimizer(0.1, 0.5).minimize(loss, var_list=variables)
        return train_op
    
    def get_noise_sample(self, batch_size):
        return np.random.uniform(low=-1, high=1, size=(batch_size, self.z_size))

    def train_one_step(self, image_batch, z):
        # train generator
        _, gen_loss_value = self.sess.run([self.gen_train_op, self.generator_loss],
                                      feed_dict={self.z: z})
        assert not np.isnan(gen_loss_value), 'Model diverged with generator NaN loss value'

        # train discriminator
        _, discrim_loss_value = self.sess.run([self.discrim_train_op, self.discriminator_loss],
                                      feed_dict={self.real_images: image_batch,
                                                 self.z: z})
        assert not np.isnan(discrim_loss_value), 'Model diverged with discriminator NaN loss value'

        return gen_loss_value, discrim_loss_value

    def train(self, steps, batch_size):
        for i in xrange(steps):
            image_batch = self.next_batch(batch_size)
            z = self.get_noise_sample(image_batch.shape[0])
            gen_loss, discrim_loss = self.train_one_step(image_batch, z)

            if i % 100 == 0:
                print 'Step %d, gen_loss: %f, discrim_loss: %f' % (i, gen_loss, discrim_loss)

    def generate(self, num_images):
        z = self.get_noise_sample(num_images)
        gen_images = self.sess.run(self.gen_images, feed_dict={self.z: z})
        return gen_images
