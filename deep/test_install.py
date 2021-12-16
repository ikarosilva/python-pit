import tensorflow as tf
from tensorflow.python.client import device_lib

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True,device_count = {'GPU': 0})) as sess:
    print (sess.run(c))

print device_lib.list_local_devices() 