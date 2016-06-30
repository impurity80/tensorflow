
import tensorflow as tf
import numpy as np

hello = tf.constant('Hello, tensorflow')

sess = tf.Session()

print sess.run(hello)

print hello

summary_writer = tf.train.SummaryWriter('.', sess.graph)