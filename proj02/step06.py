
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

y_data = [1,2,3,4,5]
x_data = [[1.0,0,3,0,5],
          [0,2,0,4,0]]

W = tf.Variable(tf.random_uniform([1,2],-1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

hypothesis = tf.matmul(W, x_data) + b

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