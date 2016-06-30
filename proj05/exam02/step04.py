import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0/(n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

train_xy = np.genfromtxt('train.csv',delimiter=',', dtype='float32')
test_xy = np.genfromtxt('test.csv', delimiter=',', dtype='float32')

x_data = train_xy[:,0:23]
y_data = train_xy[:,23:24]


X = tf.placeholder("float", [None,23])
Y = tf.placeholder("float", [None,1])

#W1 = tf.Variable(tf.zeros([23,10]))
#W2 = tf.Variable(tf.zeros([10,1]))
#b1 = tf.Variable(tf.zeros([1,10]))
#b2 = tf.Variable(tf.zeros([1,1]))

W1 = tf.get_variable("W1", shape=[23,100], initializer=xavier_init(23,100))
W2 = tf.get_variable("W2", shape=[100,1], initializer=xavier_init(100,1))
b1 = tf.get_variable("b1", shape=[1,100], initializer=xavier_init(1, 100))
b2 = tf.get_variable("b2", shape=[1,1], initializer=xavier_init(1,1))

L2 = tf.matmul(X, W1) + b1
#L2 = tf.sigmoid(tf.matrmul(X, W1) + b1)
#L2 = tf.tanh(tf.matmul(X, W1) + b1)
# hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)
hypothesis = tf.matmul(L2, W2) + b2

# hypothesis = tf.matmul(X, W) + b

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

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

# print sess.run(W), sess.run(b)


#plt.plot(y_data[:,1], result[:,1], 'ro')
#plt.show()

