import numpy as np
import tensorflow as tf
import math

a = tf.Variable([1, 2, 3], dtype=tf.float32)
b = tf.Variable([4, 5, 6], dtype=tf.float32)

ema = tf.train.ExponentialMovingAverage(decay=0.5)


def batch_run(x, is_test=False):
    _mean, _var = tf.nn.moments(x, axes=[0])

    def mean_update():
        ema_apply_oper = ema.apply([_mean, _var])
        with tf.control_dependencies([ema_apply_oper]):
            return _mean

    return tf.cond(is_test is True, lambda: (ema.average_name(_mean)), mean_update)


a_mean = batch_run(a)
b_mean = batch_run(b)
move_mean = batch_run(b, True)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print("a mean is:", session.run(a_mean))
print("b mean is:", session.run(b_mean))
print("move mean is:", move_mean)
