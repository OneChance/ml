# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# np.random.seed(1)
# tf.set_random_seed(1)

EPSILON = 0.9  # 10%的动作选择是随机的
ALPHA = 0.1  # 学习速率
GAMMA = 0.9  # 增益衰减
TRACE_DECAY = 0.9  # 选择重要性衰减


class RL(object):
    def __init__(self, n_states, actions):
        self.n_states = n_states
        self.actions = actions
        self.q_tables = self._build_q_table()

    def _build_q_table(self):
        table = pd.DataFrame(np.zeros((self.n_states, len(self.actions))), columns=self.actions)
        return table

    def choose_action(self, state):
        state_actions = self.q_tables.iloc[state, :]
        if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
            action_name = np.random.choice(self.actions)
        else:
            action_name = state_actions.argmax()
        return action_name

    def learn(self, *args):
        pass


class QLearning(RL):
    def __init__(self, n_states, actions):
        super(QLearning, self).__init__(n_states, actions)

    def learn(self, S, A, R, S_):
        # 取出当前步骤的q估计(当前步骤A动作的Q值)
        q_predict = self.q_tables.ix[S, A]
        # 计算q现实(当前步骤的该选择价值,实际上是做了选择后到达的下一步骤所获得的奖励+下一步可做出的选择能给予的最大q值)
        # R是固定的奖励,q值是不断调整从而确定的可以到达奖励的选择增益
        # 如果GAMMA接近0,说明agent只对一步能获得的奖励R比较感兴趣,对之后的选择都不太关注
        # 如果GAMMA接近1,说明agent对未来的选择增益也很在意,目光更为长远
        if S_ != 'terminal':
            q_target = R + GAMMA * self.q_tables.iloc[S_, :].max()
        else:
            q_target = R
        # 用q现实实际可以带来的价值来改善(更新)q估计,ALPHA决定了从实际与原本的估计误差中,学习多少
        # 因为这里要用标签(A)来索引,所以要使用ix混合索引
        self.q_tables.ix[S, A] += ALPHA * (q_target - q_predict)


class Sarsa(RL):
    def __init__(self, n_states, actions):
        super(Sarsa, self).__init__(n_states, actions)

    def learn(self, S, A, R, S_, A_):
        q_predict = self.q_tables.ix[S, A]
        if S_ != 'terminal':
            q_target = R + GAMMA * self.q_tables.ix[S_, A_]
        else:
            q_target = R
        self.q_tables.ix[S, A] += ALPHA * (q_target - q_predict)


class SarsaLambda(RL):
    def __init__(self, n_states, actions):
        super(SarsaLambda, self).__init__(n_states, actions)
        self.lambda_ = TRACE_DECAY
        # 选择重要程度的思想是:如果在不断的学习中,状态S下动作A经常被选中,那么这个选择对获得R是有帮助的,应当重视
        # sarsa和q-learning都是到达下一状态的增益来更新当前状态的q值
        # 而这个算法考虑了从开始到获得奖励的每一个步骤的重要性,一次行动会更新之前经历的所有状态的q值
        # 从而加速了学习
        self.eligibility_trace = self.q_tables.copy()

    def learn(self, S, A, R, S_, A_):
        q_predict = self.q_tables.ix[S, A]
        if S_ != 'terminal':
            q_target = R + GAMMA * self.q_tables.ix[S_, A_]
        else:
            q_target = R
        error = q_target - q_predict

        # 置该选择的重要度为1
        self.eligibility_trace.ix[S, :] *= 0
        self.eligibility_trace.ix[S, A] = 1

        # 这里的更新,除了更新[S,A]的q值,还更新了trace里q值不为0的[S',A']的q值(所有经历过的状态所选择的动作)
        # 但是对于其他q值更新,衰减的更厉害,也就是说离获得R越近的选择越重要
        self.q_tables += ALPHA * error * self.eligibility_trace

        # 选择的重要程度会不断衰减
        # 衰减是为了降低那些对获得R没有作用的行动的重要性,避免再次选择
        self.eligibility_trace *= GAMMA * self.lambda_


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        # 记录优先级
        self.tree = np.zeros(2 * capacity - 1)
        # 记录具体数据
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        # 写满就覆盖
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]


# DQN
class DeepQNetwork:
    def __init__(self, n_actions,
                 n_features,
                 learning_rate=0.01,
                 reword_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reword_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        # [s,a,r,s_] n_features*2是因为s和s_都有n_features个状态属性,另外2个维度是a,r
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 当前状态
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 下一步状态
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # q估计网络
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer)
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)

        # q现实网络
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer)
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)

        # 由于误差计算使用的q_target是传入的,所以只有计算q_eval的网络会学习
        self.loss = tf.losses.mean_squared_error(self.q_target, self.q_eval)
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = [[observation]]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 将q估计网络学习的参数复制给q现实网络
            self.sess.run(self.target_replace_op)

        # 随机抽取经历,打乱相关性,提升学习效率
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 分别在两个网络上跑下一个observation(state)
        q_next, q_eval_next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, -self.n_features:]
            })

        q_eval = self.sess.run(
            self.q_eval,
            feed_dict={
                self.s: batch_memory[:, :self.n_features]
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 取行为那一列,每个元素为行为的索引
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # dqn的q_target计算是在q现实网络计算出的结果中选取最大q值,这样可能会造成q值估计过高
        # q_t = reward + self.gamma * np.max(q_next, axis=1)
        # 所以double dqn是先获得q估计网络计算出的结果的最大q值的索引,然后取q现实网络计算结果中,该索引对应位置的q值
        max_q_eval_next_index = np.argmax(q_eval_next, axis=1)
        q_t = reward + self.gamma * q_next[batch_index, max_q_eval_next_index]

        # q_target只更新样本中对应动作的Q值,这样q_target-q_eval计算的误差,就只有对应动作的q值误差
        q_target[batch_index, eval_act_index] = q_t

        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={
                                    self.s: batch_memory[:, :self.n_features],  # [batch_size,1]
                                    self.q_target: q_target
                                })
        self.cost_his.append(cost)

        # 一开始由于没有记忆库的支持,大量的行为选择都是随机的,随着不断的学习,选择趋向于贪婪最优Q值
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('steps')
        plt.show()
