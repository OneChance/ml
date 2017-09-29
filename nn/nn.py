# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, x_data.shape)
ys = tf.placeholder(tf.float32, y_data.shape)
is_train = tf.placeholder(tf.bool, None)

# 添加隐藏层
l1 = tf.layers.dense(xs, 10, tf.nn.relu)
prediction = tf.layers.dense(l1, 1)

loss = tf.losses.mean_squared_error(ys, prediction)

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

plt.ion()

for i in range(1000):
    _, l, pred = session.run([train, loss, prediction], feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        plt.cla()  # 清空
        plt.scatter(x_data, y_data)
        plt.plot(x_data, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
