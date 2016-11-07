import tensorflow as tf
from mnist import MNIST
from ali import ALI
from vae import VAE
from ae import AE
from vaegan import VAEGAN

class MNIST_ALI(MNIST, ALI):
    def __init__(self):
        MNIST.__init__(self)
        ALI.__init__(self)
        self.save_dir = './log/MNIST_ALI'
        self.name = 'MNIST_ALI'

class MNIST_VAE(MNIST, VAE):
    def __init__(self):
        MNIST.__init__(self)
        VAE.__init__(self)
        self.save_dir = './log/MNIST_VAE'
        self.name = 'MNIST_VAE'

class MNIST_AE(MNIST, AE):
    def __init__(self):
        MNIST.__init__(self)
        AE.__init__(self)
        self.save_dir = './log/MNIST_AE'
        self.name = 'MNIST_AE'

class MNIST_VAEGAN(MNIST, VAEGAN):
    def __init__(self):
        MNIST.__init__(self)
        VAEGAN.__init__(self)
        self.save_dir = './log/MNIST_VAEGAN'
        self.name = 'MNIST_VAEGAN'

def get_model(name):
    with tf.Graph().as_default():
        if name == 'MNIST_ALI':
            model = MNIST_ALI()
        elif name == 'MNIST_VAE':
            model = MNIST_VAE()
        elif name == 'MNIST_AE':
            model = MNIST_AE()
        elif name == 'MNIST_VAEGAN':
            model = MNIST_VAEGAN()
        else:
            raise NotImplementedError
    return model

