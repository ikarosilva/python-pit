# Mandelbrot using tensorflow
# Source: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mandelbrot/index.md
# https://github.com/tobigithub/tensorflow-deep-learning/wiki

# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from cStringIO import StringIO
import scipy.ndimage as nd
import matplotlib.pyplot as plt 


with tf.Session() as sess:
    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    
    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    Z = X+1j*Y
    
    xs = tf.constant(Z.astype("complex64"))
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, "float32"))
    
    tf.global_variables_initializer()
    # Compute the new values of z: z^2 + x
    zs_ = zs*zs + xs
    
    # Have we diverged with this new value?
    not_diverged = tf.abs(zs_) < 4
    
    # Operation to update the zs and the iteration count.
    #
    # Note: We keep computing zs after they diverge! This
    #       is very wasteful! There are better, if a little
    #       less simple, ways to do this.
    #
    step = tf.group(
      zs.assign(zs_),
      ns.assign_add(tf.cast(not_diverged, "float32"))
      )
    
    for i in range(200): step.run()
    plt.imshow(ns.eval())

### END