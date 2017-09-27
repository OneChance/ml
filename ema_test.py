# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 3])
on_train = tf.placeholder(tf.bool)


# 这个方法模拟一个网络层里的操作,在算出z值后,做batch-normalization,但是对于测试集,
# 不能使用分批计算的均值和方差,所以考虑训练集中每批数据计算的均值和方差,通过移动平均值来估计
# 测试集的均值和方差(不能使用测试集独立计算的均值和方差,这么做是为了保证测试集和训练集来自同一分布)
def batch_run(_x, _on_train):
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    _mean, _var = tf.nn.moments(x, axes=[0, 1])
    tf.summary.histogram('batch_mean', _mean)

    def mean_update():
        ema_apply_oper = ema.apply([_mean, _var])
        with tf.control_dependencies([ema_apply_oper]):
            return tf.identity(_mean)

    # 此处必须把包含ema.apply()的方法写在true_fn的位置,也就是在构建flow的时候,
    # 必须先执行apply()再执行average()
    return tf.cond(_on_train, mean_update, lambda: ema.average(_mean))


mean = batch_run(x, on_train)
tf.summary.scalar('mean', mean)

init = tf.global_variables_initializer()
session = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", session.graph)

session.run(init)
for i in range(1, 3):
    bx = np.array([[i * 1, i * 2, i * 3]])
    train_mean = session.run(mean, feed_dict={x: bx, on_train: True})
    print(train_mean)

test_mean = session.run(mean, feed_dict={x: [[10, 20, 30]], on_train: False})
print(test_mean)

# 输出:
# 2.0  2.0 = (1+2+3)/3
# 4.0  4.0 = (2+4+6)/3
# 2.5  训练集估计的移动平均值:_mean作为Tensor被初始化为0.0  2.5 = (0.0*0.5 + 2*0.5)*0.5+4*0.5
