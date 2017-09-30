# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

np.random.seed(2)

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
