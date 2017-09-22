import tensorflow as tf
import numpy as np


def add_layer(n_layer, inputs, in_size, out_size, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([in_size, out_size]), name='weights')
            tf.summary.histogram(layer_name + '/weights', W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')
            tf.summary.histogram(layer_name + '/biases', b)
        with tf.name_scope('z'):
            z = tf.matmul(inputs, W) + b
        if activation_function is None:
            a = z
        else:
            a = activation_function(z)
        return a


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(1, xs, 1, 10, activation_function=tf.nn.relu)
l2 = add_layer(2, l1, 10, 10, activation_function=tf.nn.relu)
prediction = add_layer(3, l2, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
session = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", session.graph)
session.run(init)

for i in range(1000):
    session.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = session.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
