# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import time
import os

np.random.seed(2)

N_STATE = 6  # 6个状态
ACTIONS = ["left", "right"]  # 可选择的动作
EPSILON = 0.9  # 10%的动作选择是随机的
ALPHA = 0.1  # 学习速率
LAMBDA = 0.9  # 增益衰减
MAX_EPISODES = 13  # 训练13代
FRESH_TIME = 0.3  # 画面刷新时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATE - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATE - 1) + ['T']
    if S == 'terminal':
        interaction = '第%s代：找到宝藏用了%s步' % (episode + 1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(2)
        print('\r')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            # 选择动作
            A = choose_action(S, q_table)
            # 获得当前步骤执行该动作后得到的下一步骤以及到达下一步给予的奖励
            S_, R = get_env_feedback(S, A)
            # 取出当前步骤的q估计(当前步骤所有动作的Q值)
            q_predict = q_table.ix[S, A]
            # 计算q现实(当前步骤的该选择价值,实际上是做了选择后到达的下一步骤所获得的奖励+下一步可做出的选择能给予的最大q值)
            # R是固定的奖励,q值是不断调整从而确定的可以到达奖励的选择增益
            # 如果LAMBDA接近0,说明agent只对一步能获得的奖励R比较感兴趣,对之后的选择都不太关注
            # 如果LAMBDA接近1,说明agent对未来的选择增益也很在意,目光更为长远
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            # 用q现实实际可以带来的价值来改善(更新)q估计,ALPHA决定了从实际与原本的估计误差中,学习多少
            # 因为这里要用标签(A)来索引,所以要使用ix混合索引
            q_table.ix[S, A] += ALPHA * (q_target - q_predict)

            S = S_
            step_counter += 1
            update_env(S, episode, step_counter)
    return q_table


q_table_res = rl()
print(q_table_res)
