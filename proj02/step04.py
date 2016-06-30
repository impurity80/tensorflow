
import tensorflow as tf
import numpy as np

sess = tf.Session()

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))


hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%20==0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)

summary_writer = tf.train.SummaryWriter('events', sess.graph)

print sess.run(hypothesis, feed_dict={X:5})