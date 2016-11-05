import mnist
import ali

class MNIST_ALI(mnist.MNIST, ali.ALI):
    def __init__(self):
        mnist.MNIST.__init__(self)
        ali.ALI.__init__(self)
        self.save_dir = './log/MNIST_ALI'
        self.name = 'MNIST_ALI'

