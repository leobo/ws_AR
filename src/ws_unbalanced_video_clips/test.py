import tensorflow as tf
import numpy as np

a = [1,2,3]
b = [1,2,3,4,5,6]
c = [1,1,1]

sess = tf.Session()
d = sess.run(tf.nn.convolution(c, a, strides=3, padding="same"))
e = sess.run(tf.nn.convolution(c, b, strides=6, padding="same"))
print()