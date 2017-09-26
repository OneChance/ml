# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size])) + 0.1
    z = tf.matmul(inputs, W) + b

    # batch normalization
    batch_mean, batch_var = tf.nn.moments(z, axes=[0])  # [0] for batch. for image axes=[0,1,2] [batch,height,width]
    scale = tf.Variable(tf.ones([out_size]))
    offset = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001

    def mean_var_with_update():
        # 保存每一批的均值和方差
        ema_apply = ema.apply([batch_mean, batch_var])
        # 控制必须在保存之后
        with tf.control_dependencies([ema_apply]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = mean_var_with_update()

    z = tf.nn.batch_normalization(z, mean=mean, variance=var, offset=offset, scale=scale, variance_epsilon=epsilon)

    if activation_function is None:
        a = z
    else:
        a = activation_function(z)
    return a


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
l2 = add_layer(l1, 10, 10, activation_function=tf.nn.sigmoid)
prediction = add_layer(l2, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# train = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    session.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(session.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = session.run(prediction, feed_dict={xs: x_data, ys: y_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
