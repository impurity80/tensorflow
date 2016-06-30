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

W1 = tf.Variable(tf.random_uniform([23,1],-1.0,1.0))
b1 = tf.Variable(tf.random_uniform([1,1],-1.0,1.0))

model = tf.matmul(X, W1) + b1

saver = tf.train.Saver()

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver.restore(sess, './model.ckpt')

result_train = sess.run(model, feed_dict={X:x_data, Y:y_data})

plt.plot(train_xy[:,23:24], result_train[:,0], 'ro')
plt.plot([-1,1],[-1,1])
plt.show()

result_test = sess.run(model, feed_dict={X:test_xy[:,0:23], Y:test_xy[:,23:24]})

plt.plot(test_xy[:,23:24], result_test[:,0], 'ro')
plt.plot([-1,1],[-1,1])
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
