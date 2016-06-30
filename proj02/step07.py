
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.zeros([1,3]))
b = tf.Variable(tf.zeros([1]))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis-y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step%20==0:
        print step, sess.run(cost), sess.run(W), sess.run(b)

summary_writer = tf.train.SummaryWriter('events', sess.graph)