# -*- coding:utf-8 -*-

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from env import Env
import os
import shutil
import matplotlib.pyplot as plt

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 100
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001
LR_C = 0.001
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = Env()
N_S = env.n_features
N_A = len(env.actions)

"""
a3c其实就是一种分布式的actor-critic,这种特性可以打乱学习样本的相关性
"""


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')

                # 实际运行中获得的奖励
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                # actor计算的a_prob,critic计算的奖励,以及两个网络的参数
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                # 计算critic的td_error
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    # 获取中央大脑的参数
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    # 向中央大脑提交学习结果，在中央大脑的参数上直接应用学习到的梯度
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 20, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 10, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: [[s]]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        return action


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = Env()
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        episode = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and episode < MAX_GLOBAL_EP:
            self.env.reset()
            total_step = 0
            while True:
                if self.name == 'W_0':
                    self.env.render(episode, total_step)
                a = self.AC.choose_action(self.env.S)
                s_, r, done = self.env.step(a)

                buffer_s.append(self.env.S)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0:
                    # 每UPDATE_GLOBAL_ITER步学习一次
                    v_s_ = SESS.run(self.AC.v, {self.AC.s: [[s_]]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # 翻转buffer,以便进行奖励衰减计算
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                self.env.S = s_
                total_step += 1
                if done:
                    if self.name == 'W_0':
                        self.env.render(episode, total_step)
                    episode += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # 中央大脑只用来保存参数
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []
        # 创建worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)