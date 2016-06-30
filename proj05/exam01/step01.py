import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

xy = np.loadtxt('train1.txt', unpack=True, dtype='float32')

print xy.shape

x_data = xy[0:5]
y_data = xy[5:7]

print x_data.shape, y_data.shape
print y_data

W = tf.Variable(tf.zeros([2,5]))
b = tf.Variable(tf.zeros([2,1]))

hypothesis = tf.matmul(W, x_data) + b

cost = tf.reduce_mean(tf.square(hypothesis-y_data))

a = tf.Variable(0.1)
#optimizer = tf.train.GradientDescentOptimizer(a)
optimizer = tf.train.AdamOptimizer(a)
train = optimizer.minimize(cost)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for step in xrange(20001):
    sess.run(train)
    if step%200==0:
        print step, sess.run(cost), sess.run(W), sess.run(b)

result = sess.run(hypothesis)

plt.plot(y_data[0], result[0], 'ro')
plt.show()

plt.plot(y_data[1], result[1], 'ro')
plt.show()

summary_writer = tf.train.SummaryWriter('events', sess.graph)