# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

alpha = 0.01
training_iters = 100000
display_step = 20

n_inputs = 28  # pixels in one row
n_steps = 28  # rows of images pixels,each row is input of time t.
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


def RNN(X):
    # 隐藏层输入为128个神经元，时序为28,共128个样本
    X_in = tf.reshape(X, [-1, n_steps, n_inputs])  # 28row,128col,128depth

    # 设置细胞
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units)

    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=None, dtype=tf.float32, time_major=False)

    results = tf.layers.dense(outputs[:, -1, :], n_classes)

    return results


prediction = RNN(x)
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
train = tf.train.AdamOptimizer(alpha).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(prediction, axis=1))[1]

session = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
session.run(init)

step = 0
batch_size = 64

while step * batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])  # 28row,28col,128depth
    session.run(train, feed_dict={
        x: batch_xs,
        y: batch_ys
    })
    if step % display_step == 0:
        print(session.run(accuracy, feed_dict={
            x: mnist.test.images.reshape([-1, n_steps, n_inputs]),
            y: mnist.test.labels
        }))
    step = step + 1
