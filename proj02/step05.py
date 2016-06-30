
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

y_data = [1,2,3,4,5]
x1_data = [1,0,3,0,5]
x2_data = [0,2,0,4,0]

W1 = tf.Variable(tf.zeros([1]))
W2 = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))

hypothesis = W1*x1_data + W2*x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis-y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step%20==0:
        print step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)

summary_writer = tf.train.SummaryWriter('events', sess.graph)