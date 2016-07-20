import numpy as np
from sklearn.datasets import fetch_mldata

def shuffle_in_unison_inplace(a, b):
    assert(len(a) == len(b))
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    
    # Normalize between 0 and 1
    mnist.data = mnist.data.astype(np.float32) / 255.0

    # Random shuffle
    mnist.data, mnist.target = shuffle_in_unison_inplace(mnist.data, mnist.target)

    return mnist.data, mnist.target