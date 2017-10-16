# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

np.random.seed(2)
tf.set_random_seed(2)


class Actor(object):
    """
    Actor其实就是一个policy gradient,
    只不过他乘上行为选择概率的那个值（td_error）是critics告诉他的,
    td_error是选择的行为
    而不是自己通过一个episode记录下来的奖励集合计算的
    """

    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = [[s]]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = [[s]]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):
    """
    critic相当于一个q-learning,
    不断修正q值的计算
    用下一步奖励加上本步奖励的价值提升（td_error）来指导actor对行为的选择
    """

    def __init__(self, sess, n_features, lr=0.01, gamma=0.9):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            # self.r+gamma*self.v类似于q实际,self.v类似于q估计,
            # 相当于是q-learning的计算
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = [[s]], [[s_]]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})

        return td_error


class ActorCritic(object):
    """
    actor_critic将两者结合,
    即有actor可以基于连续无限动作做出选择（q-learning只能基于离散有限的动作选择,因为有max选择）的优点,
    又有q-learning单步学习的特性来提升学习效率,
    但是由于是通过连续的样本来学习,所以存在缺陷
    """

    def __init__(self, n_actions, n_features, lr=0.001, gamma=0.9):
        sess = tf.Session()

        self.actor = Actor(sess, n_features=n_features, n_actions=n_actions, lr=lr)
        self.critic = Critic(sess, n_features=n_features, lr=lr, gamma=gamma)

        sess.run(tf.global_variables_initializer())
