import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

def normalize_list(input_list, min, max):
    if max > min:
        return (input_list-min)/(max-min)-0.5
    else :
        return input_list

def denormalize_list(input_list, min, max):
    if max > min:
        return (input_list+0.5)*(max-min)+min
    else:
        return input_list

def normalize(input_xy, minmax):
    output_xy = []
    for (y, m) in zip(np.transpose(input_xy), minmax):
        output_xy.append(normalize_list(y, m[0], m[1]))
    return np.transpose(output_xy)

def denormalize(input_xy, minmax):
    output_xy = []
    for (y, m) in zip(np.transpose(input_xy), minmax):
        output_xy.append(denormalize_list(y, m[0], m[1]))
    return np.transpose(output_xy)

train_xy = np.genfromtxt('train.csv',delimiter=',', dtype='float32')
test_xy = np.genfromtxt('test.csv', delimiter=',', dtype='float32')
minmax = np.genfromtxt('MINMAX.csv', delimiter=',', dtype='float32')
train_xy = normalize(train_xy, minmax)
test_xy = normalize(test_xy, minmax)

#-------------------regression ----------------------

x_data = train_xy[:,0:23]
y_data = train_xy[:,23:24]

X = tf.placeholder("float", [None,23])
Y = tf.placeholder("float", [None,1])

W1 = tf.Variable(tf.random_uniform([23,10],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([10,3],-1.0,1.0))
W3 = tf.Variable(tf.random_uniform([3,2],-1.0,1.0))
W4 = tf.Variable(tf.random_uniform([2,1],-1.0,1.0))

b1 = tf.Variable(tf.random_uniform([1,10],-1.0,1.0))
b2 = tf.Variable(tf.random_uniform([1,3],-1.0,1.0))
b3 = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
b4 = tf.Variable(tf.random_uniform([1,1],-1.0,1.0))

W = tf.Variable(tf.ones([23,1]))
b = tf.Variable(tf.ones([1,1]))
#L2 = tf.matmul(X, W1) + b1
#L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
#L1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)
#hypothesis = tf.sigmoid(tf.matmul(L2, W3) + b3)


#L1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
#L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
# hypothesis = tf.nn.relu(tf.matmul(L2, W3) + b3)
#hypothesis = tf.nn.sigmoid(tf.matmul(L2, W3) + b3)

L1 = tf.nn.tanh(tf.matmul(X, W1) + b1)
L2 = tf.nn.tanh(tf.matmul(L1, W2) + b2)
# hypothesis = tf.nn.tanh(tf.matmul(L2, W3) + b3)
L3 = tf.nn.tanh(tf.matmul(L2, W3) + b3)
hypothesis = tf.nn.tanh(tf.matmul(L3, W4) + b4)

# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)
#hypothesis = tf.sigmoid(tf.matmul(L2, W3) + b3)
#hypothesis = tf.matmul(L2, W2) + b2

#L1 = tf.tanh(tf.matmul(X, W1) + b1)
#L2 = tf.tanh(tf.matmul(L1, W2) + b2)
#hypothesis = tf.tanh(tf.matmul(L2, W3) + b3)

#cost = tf.reduce_mean(tf.square(hypothesis-Y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
#cost = tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)
#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(hypothesis, Y))
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis)))
cost = tf.reduce_mean(tf.abs(hypothesis-Y))

optimizer = tf.train.AdamOptimizer(0.05)
# optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for step in xrange(20001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%200==0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data})

result_train = sess.run(hypothesis, feed_dict={X:x_data, Y:y_data})

plt.plot(train_xy[:,23:24], result_train[:,0], 'ro')
plt.plot([0,1],[0,1])
plt.show()

result_test = sess.run(hypothesis, feed_dict={X:test_xy[:,0:23], Y:test_xy[:,23:24]})

plt.plot(test_xy[:,23:24], result_test[:,0], 'ro')
plt.plot([0,1],[0,1])
plt.show()


train_xy = denormalize(train_xy, minmax)
test_xy = denormalize(test_xy, minmax)
m1 = minmax[23,0]
m2 = minmax[23,1]
result_train = denormalize_list(result_train, minmax[23,0], minmax[23,1])
result_test = denormalize_list(result_test, minmax[23,0], minmax[23,1])

plt.plot(train_xy[:,23:24], result_train[:,0], 'ro')
plt.plot([m1,m2],[m1,m2])
plt.show()

plt.plot(test_xy[:,23:24], result_test[:,0], 'ro')
plt.plot([m1,m2],[m1,m2])
plt.show()