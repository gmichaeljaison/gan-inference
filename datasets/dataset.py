import tensorflow as tf


class GanDataset:
    """
    Dataset interface for GAN framework
    """
    def __init__(self, z_size, batch_size=100):
        """
        Constructor that initializes placeholders and weight initializer

        :param data: Tensorflow data object
        :param z_size: encoding size
        """
        self.data = self.read_data()
        self.batch_size = batch_size
        self.z_size = z_size
        self.x_size = self.data.train.images.shape[1]

        self.x = tf.placeholder(tf.float32, shape=[None, self.x_size])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_size])
        self.weights_initializer = lambda: tf.truncated_normal_initializer(stddev=0.01)

        self.activation = tf.nn.relu

    def read_data(self):
        pass

    def encoder(self, x):
        pass

    def generator(self, z):
        pass

    def discriminator(self, x):
        pass

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return self.data.train.next_batch(batch_size)[0]
