
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

c = a+b

sess = tf.Session()
print c
print sess.run(a+b)
print sess.run(a*b)

summary_writer = tf.train.SummaryWriter('.', sess.graph)