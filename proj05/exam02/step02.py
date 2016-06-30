import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

train_xy = np.genfromtxt('train.csv',delimiter=',', dtype='float32')
test_xy = np.genfromtxt('test.csv', delimiter=',', dtype='float32')

# train_xy = np.transpose(train_xy)

x_data = train_xy[:,0:23]
y_data = train_xy[:,23:24]

print x_data.shape, y_data.shape

print y_data

#W = tf.Variable(tf.zeros([23,2]))
#b = tf.Variable(tf.zeros([1,2]))
W = tf.Variable(tf.zeros([23,1]))
b = tf.Variable(tf.zeros([1,1]))

hypothesis = tf.matmul(x_data, W) + b
sq = tf.square(hypothesis-y_data)
cost = tf.reduce_mean(tf.square(hypothesis-y_data))

a = tf.Variable(0.1)
optimizer = tf.train.AdamOptimizer(a)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for step in xrange(20001):
    sess.run(train)
    if step%200==0:
        print step, sess.run(cost)

result = sess.run(hypothesis)

plt.plot(y_data[:,0], result[:,0], 'ro')
plt.show()

with open('result.csv','w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|')
    for x in zip(y_data[:,0],result[:,0]):
        writer.writerow(x)
    f.close()

#plt.plot(y_data[:,1], result[:,1], 'ro')
#plt.show()

