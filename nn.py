import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size])) + 0.1
    z = tf.matmul(inputs, W) + b
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
l2 = add_layer(l1, 10, 3, activation_function=tf.nn.relu)
prediction = add_layer(l2, 3, 1, activation_function=None)

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
