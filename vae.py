import tensorflow as tf
import numpy as np
import os.path

from model_base import Model_Base

class VAE(Model_Base):

    def __init__(self):
        mu, log_sigma, self.encoder_features = self.encoder(self.real_images)
        eps = tf.truncated_normal([tf.shape(mu)[0], self.z_size])
        self.z_encoded = mu + eps*tf.exp(log_sigma)

        self.gen_images = self.generator(self.z_encoded)

        output_sigma = 1.0
        #reconstruction_loss = tf.reduce_sum(tf.square(self.gen_images - self.real_images), reduction_indices=[1,2,3])
        #reconstruction_loss = 0.5*tf.reduce_mean(reconstruction_loss) / output_sigma**2
        #KL_loss = tf.reduce_sum(1 + 2*log_sigma - mu**2 - tf.exp(2*log_sigma), reduction_indices=[1])
        #KL_loss = -0.5*tf.reduce_mean(KL_loss)
        self.reconstruction_loss = tf.reduce_mean(tf.square(self.gen_images - self.real_images))
        self.KL_loss = -output_sigma**2 * self.z_size/self.x_size*tf.reduce_mean(1 + 2*log_sigma - mu**2 - tf.exp(2*log_sigma))
        self.loss = self.reconstruction_loss + self.KL_loss

        self.train_op = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(self.loss)

        Model_Base.__init__(self)
        self.init_op = tf.initialize_all_variables()

    def train_one_step(self, image_batch):
        _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict={self.real_images: image_batch})
        assert not np.isnan(loss_value), 'Model diverged with NaN loss value'

        return loss_value

    def train(self, steps, batch_size):
        if tf.gfile.Exists(self.save_dir):
            tf.gfile.DeleteRecursively(self.save_dir)
        tf.gfile.MakeDirs(self.save_dir)

        self.sess.run(self.init_op)
        for i in xrange(steps):
            image_batch = self.next_batch(batch_size)
            # sampled z is not actually used, but must be fed in because z_sampled is a placeholder
            loss = self.train_one_step(image_batch)

            if i % 10 == 0:
                print 'Step %d, loss = %f' % (i, loss)

            if i % 1000 == 0 or (i+1) == steps:
                self.saver.save(self.sess, os.path.join(self.save_dir, self.name), global_step=i)

    # redefine this because VAE doesn't use z_sampled, only z_encoded
    def generate(self, z):
        gen_images = self.sess.run(self.gen_images, feed_dict={self.z_encoded: z})
        return gen_images
