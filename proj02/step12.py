import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

xy = np.loadtxt('train4.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:-1])
y_data = np.transpose(xy[-1])

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

W1 = tf.Variable(tf.random_uniform([2,10], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([10,1], -1.0, 1.0), name='Weight2')
b1 = tf.Variable(tf.random_uniform([10], -1.0, 1.0), name="Bias1")
b2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Bias2")
#W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))

#h = tf.matmul(X,W)
#h = tf.matmul(W, x_data)
#hypothesis = tf.div(1., 1.+tf.exp(-h))

with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1)+b1)

with tf.name_scope("layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    cost_summ = tf.scalar_summary("cost", cost)

with tf.name_scope("train") as scope:
    a = tf.Variable(0.01)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

w1_hist = tf.histogram_summary("weight1", W1)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("events", sess.graph)

    sess.run(init)

    for step in xrange(1000):
     #   sess.run(train, feed_dict={X:x_data,Y:y_data})
        summary, _ =sess.run([merged,train], feed_dict={X:x_data, Y:y_data})

        writer.add_summary(summary, step)
        if step%200==0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)

    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print '------------------'
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy:", accuracy.eval({X:x_data, Y:y_data})


# summary_writer = tf.train.SummaryWriter('events', sess.graph)
