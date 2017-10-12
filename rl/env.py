# -*- coding:utf-8 -*-

import time

N_STATE = 6  # 状态数
ACTIONS = ["left", "right"]  # 可选择的动作
FRESH_TIME = 0.3  # 画面刷新时间


class Env:
    def __init__(self):
        self.states = N_STATE
        self.actions = ACTIONS
        # 状态的维度(这里用1维的整数即可表示每一种可能的状态:0-5),用于确定神经网络输入参数维度
        self.n_features = 1

    def reset(self):
        self.S = 0

    def step(self, A):
        done = False
        # if A == 'right':
        if A == 1:  # for dqn
            if self.S == self.states - 2:
                # S_ = 'terminal'
                S_ = -1  # for dqn
                R = 1
                done = True
            else:
                S_ = self.S + 1
                R = 0
        else:
            R = 0
            if self.S == 0:
                S_ = self.S
            else:
                S_ = self.S - 1
        return S_, R, done

    def render(self, episode, step_counter):
        env_list = ['-'] * (self.states - 1) + ['T']
        # if self.S == 'terminal':
        if self.S == -1:  # for dqn
            interaction = '第%s代：找到宝藏用了%s步' % (episode + 1, step_counter)
            print('\r{}'.format(interaction))
            time.sleep(2)
            print('\r')
        else:
            env_list[self.S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(FRESH_TIME)
