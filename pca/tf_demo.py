#  python pca/tf_demo.py 784 12
#dims: The number of dimensions to reduce to (from 784)
#index: The index of the image to show at the end
import numpy as np
import tensorflow as tf
import sys

import matplotlib.pyplot as plt

def normalize(X):
    means = np.mean(X,axis=0)
    tmp = np.subtract(X,means)
    return tmp,means

def denormalize(Rn,means):
    return np.add(Rn,means)

def showim(lin):
    twodarr = np.array(lin).reshape((28,28))
    plt.imshow(twodarr,cmap='gray')
    plt.show()

if __name__ == '__main__':
    dims = 50
    index = 0
    if (len(sys.argv) >1):
        dims = int(sys.argv[1])
    if (len(sys.argv) > 2):
        index = int(sys.argv[2])
    X = np.load('../data/testX.npy')
    Xn,means = normalize(X)
    Cov = np.matmul(np.transpose(Xn),Xn)
    Xtf = tf.placeholder(tf.float32,shape=[X.shape[0],X.shape[1]])
    Covtf = tf.placeholder(tf.float32,shape=[Cov.shape[0],Cov.shape[1]])
    stf,utf,vtf = tf.svd(Covtf)
    tvtf = tf.slice(vtf,[0,0],[784,dims])
    Ttf = tf.matmul(Xtf,tvtf)
    Rtf = tf.matmul(Ttf,tvtf,transpose_b=True)
    with tf.Session() as sess:
        Rn = sess.run(Rtf,feed_dict={Xtf:Xn,Covtf:Cov})
    R = denormalize(Rn,means)
    showim(X[index])
    showim(R[index])
