

import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.mul(a,b)

sess = tf.Session()

print sess.run(add, feed_dict={a:2, b:3})

summary_writer = tf.train.SummaryWriter('.', sess.graph)