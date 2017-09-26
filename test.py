import numpy as np
import tensorflow as tf
import math

x = tf.placeholder(tf.float32, [None, 3])
is_test = tf.placeholder(tf.bool)


def batch_run(_x, _is_test):
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    _mean, _var = tf.nn.moments(x, axes=[0, 1])

    def mean_update():
        ema_apply_oper = ema.apply([_mean, _var])
        with tf.control_dependencies([ema_apply_oper]):
            return tf.identity(_mean)

    def m_mean():
        return ema.average(_mean) or 99.0

    return tf.cond(_is_test, m_mean, mean_update)


'''
ema = tf.train.ExponentialMovingAverage(decay=0.5)
_mean, _var = tf.nn.moments(x, axes=[0, 1])

ema_apply_oper = ema.apply([_mean, _var])

with tf.control_dependencies([ema_apply_oper]):
    mean = tf.cond(is_test, lambda: ema.average(_mean), lambda: _mean)
'''

mean = batch_run(x, is_test)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
for i in range(1, 3):
    bx = np.array([[i * 1, i * 2, i * 3]])
    print(session.run(mean, feed_dict={x: bx, is_test: False}))
print(session.run(mean, feed_dict={x: [[10, 20, 30]], is_test: True}))
