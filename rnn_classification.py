# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

alpha = 0.001
training_iters = 100000
display_step = 20

n_inputs = 28  # pixels in one row
n_steps = 28  # rows of images pixels,each row is input of time t.
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


def RNN(X, _weights, _biases):
    _batch_size = tf.shape(X)[0]

    # 因为是一行一行的循环训练,所以单个样本是一行28个像素，这里转换一下然后计算z
    # 作为隐藏层输入
    X = tf.reshape(X, [-1, n_inputs])  # 128*28row,28col
    X_in = tf.matmul(X, _weights['in']) + _biases['in']

    # 隐藏层输入为128个神经元，时序为28,共128个样本
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # 28row,128col,128depth

    # 设置细胞
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # 初始化细胞状态（此处有c_state,m_state）,暂时理解如下
    # c_state是原输出,m_state是经过门（决定丢弃什么信息，记住什么信息）控制后的输出
    # 将会同时考虑c_state,m_state,并最终用m_state取代c_state
    init_state = lstm_cell.zero_state(_batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # results = tf.matmul(states[1], weights['out']) + biases['out']
    # 这里通过转置把时序转到主要轴上,然后拆分成list,list的每一个元素是128*128的矩阵
    # 第一个128是batch_size
    # 第二个128是n_hidden_units
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # 取最后一个输出
    results = tf.matmul(outputs[-1], _weights['out']) + _biases['out']

    return results


prediction = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train = tf.train.AdamOptimizer(alpha).minimize(cost)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    step = 0
    batch_size = 128

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
