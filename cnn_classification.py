# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
np.random.seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
is_train = tf.placeholder(tf.bool, None)

x_image = tf.reshape(xs, [-1, 28, 28, 1])

# 32个5*5过滤器做卷积 28*28*32
conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
# 对2*2像素做最大池化 14*14*32
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

# 14*14*64
conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
# 7*7*64
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
# 池化输出扁平化,作为全连接层的输入
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
layer1 = tf.layers.dense(flat, 256, activation=tf.nn.relu)
# dropout
layer1 = tf.layers.dropout(layer1, rate=0.5, training=is_train)
# batch-normalization(此处灰度图片数据集输入已经是0到1之间的数,所以做batch-normalization效果反而不好)
# layer1 = tf.layers.batch_normalization(layer1, training=is_train)
layer2 = tf.layers.dense(layer1, 300, activation=tf.nn.relu)
# dropout
layer2 = tf.layers.dropout(layer2, rate=0.5, training=is_train)
# batch-normalization
# layer1 = tf.layers.batch_normalization(layer1, training=is_train)
# 下面计算交叉熵时,会应用softmax,所以此层不使用激励函数
prediction = tf.layers.dense(layer2, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)

train = tf.train.AdamOptimizer(0.001).minimize(loss)

# 测试样本和预测结果都是m*10矩阵 argmax可以找到axis1(即矩阵的列)上最大的值的下标(这个位置即为预测的数字)
# 返回值为accuracy,update_op(更新accuracy的操作)
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, axis=1), predictions=tf.argmax(prediction, axis=1))[1]

session = tf.Session()
# local用于初始化accuracy更新操作里的本地变量
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
session.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train, feed_dict={xs: batch_xs, ys: batch_ys, is_train: True})
    if i % 50 == 0:
        accuracy_, loss_ = session.run([accuracy, loss],
                                       feed_dict={xs: mnist.test.images[:300], ys: mnist.test.labels[:300],
                                                  is_train: False})
        print(accuracy_, loss_)
