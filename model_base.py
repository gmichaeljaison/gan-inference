import tensorflow as tf
import numpy as np

class Model_Base:

    def __init__(self):
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess = tf.Session()

    def get_noise_sample(self, batch_size):
        return np.random.normal(scale=1, size=(batch_size, self.z_size))

    def generate(self, z):
        gen_images = self.sess.run(self.gen_images, feed_dict={self.z_sampled: z})
        return gen_images

    def inference(self, x):
        z = self.sess.run(self.z_encoded, feed_dict={self.real_images: x})
        return z

    def get_encoder_features(self, x):
        features = self.sess.run(self.encoder_features, feed_dict={self.real_images: x})
        return features

    def restore(self):
        ckpt = tf.train.latest_checkpoint(self.save_dir)
        self.saver.restore(self.sess, ckpt)

