import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

train_xy = np.genfromtxt('train.csv',delimiter=',', dtype='float32')
test_xy = np.genfromtxt('test.csv', delimiter=',', dtype='float32')

x_data = train_xy[:,0:23]
y_data = train_xy[:,23:24]

X = tf.placeholder("float", [None,23])
Y = tf.placeholder("float", [None,1])

W = tf.Variable(tf.zeros([23,1]))
b = tf.Variable(tf.zeros([1,1]))

hypothesis = tf.matmul(X, W) + b
sq = tf.square(hypothesis-Y)
cost = tf.reduce_mean(tf.square(hypothesis-Y))

a = tf.Variable(0.5)
optimizer = tf.train.AdamOptimizer(a)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for step in xrange(80001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%200==0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data})

result_train = sess.run(hypothesis, feed_dict={X:x_data, Y:y_data})

plt.plot(train_xy[:,23:24], result_train[:,0], 'ro')
plt.show()

result_test = sess.run(hypothesis, feed_dict={X:test_xy[:,0:23], Y:test_xy[:,23:24]})

plt.plot(test_xy[:,23:24], result_test[:,0], 'ro')
plt.show()

with open('train_result.csv','w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|')
    for x in zip(train_xy[:,23:24][:,0],result_train[:,0],train_xy[:,23:24][:,0]-result_train[:,0]):
        writer.writerow(x)
    f.close()

with open('test_result.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|')
    for x in zip(test_xy[:,23:24], result_test[:,0]):
        writer.writerow(x)
    f.close()

data = []
data.append(sess.run(cost, feed_dict={X:train_xy[:,0:23], Y:train_xy[:,23:24]}))
data.append(sess.run(cost, feed_dict={X:test_xy[:,0:23], Y:test_xy[:,23:24]}))
np.savetxt('summary.txt', data)

print sess.run(W), sess.run(b)


#plt.plot(y_data[:,1], result[:,1], 'ro')
#plt.show()

