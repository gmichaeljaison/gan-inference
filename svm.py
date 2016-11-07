import numpy as np
from sklearn import svm
import sys
import pickle

def random_samples(data, num):
    N = data[0].shape[0]
    indices = np.random.permutation(N)[:num]
    return (data[0][indices], data[1][indices])

with open('./features_datasets/%s.pkl' % sys.argv[1], 'rb') as f:
    dataset = pickle.load(f)

model = svm.LinearSVC(C=1.0, loss='hinge', penalty='l2')
K = 50
avg_err = 0
for i in xrange(K):
    X, y = random_samples(dataset['train'], 100)
    model.fit(X, y)
    avg_err += 1 - model.score(*dataset['validation'])
avg_err /= K
print 'Average validation error: %f%%' % (100*avg_err)

