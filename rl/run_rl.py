# -*- coding:utf-8 -*-
from env import Env
from rl_brain import QLearning, Sarsa, SarsaLambda, DeepQNetwork

MAX_EPISODES = 300  # 训练13代


# q-learning
# def q_train():
#     for episode in range(MAX_EPISODES):
#         step_counter = 0
#         env.reset()
#         env.render(episode, step_counter)
#         while True:
#             # 选择动作
#             A = learning.choose_action(env.S)
#             # 获得当前步骤执行该动作后得到的下一步骤以及到达下一步给予的奖励
#             S_, R, done = env.step(A)
#             # 学习
#             learning.learn(env.S, A, R, S_)
#             # 更新环境状态
#             env.S = S_
#             # 记录步数
#             step_counter += 1
#             # 渲染环境
#             env.render(episode, step_counter)
#
#             if done:
#                 break


# def s_train():
#     for episode in range(MAX_EPISODES):
#         step_counter = 0
#         env.reset()
#         env.render(episode, step_counter)
#         A = learning.choose_action(env.S)
#         while True:
#             S_, R, done = env.step(A)
#             if not done:
#                 A_ = learning.choose_action(S_)
#             learning.learn(env.S, A, R, S_, A_)
#             env.S = S_
#             A = A_
#             step_counter += 1
#             env.render(episode, step_counter)
#             if done:
#                 break


# DQN
def dqn_train():
    step = 0
    for episode in range(MAX_EPISODES):
        step_counter = 0
        env.reset()
        env.render(episode, step_counter)
        while True:
            A = learning.choose_action(env.S)
            S_, R, done = env.step(A)

            # 保存到记忆库
            learning.store_transition(env.S, A, R, S_)

            # 当记忆库有积累再开始学习
            if (step > 200) and (step % 5 == 0):
                learning.learn()

            env.S = S_
            step_counter += 1
            env.render(episode, step_counter)

            step += 1

            if done:
                break


env = Env()

# learning = QLearning(env.states, env.actions)
# q_train()
# learning = Sarsa(env.states, env.actions)
# learning = SarsaLambda(env.states, env.actions)
# s_train()
# print(learning.q_tables)

learning = DeepQNetwork(env.actions,
                        env.n_features,
                        learning_rate=0.01,
                        reword_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=200,
                        memory_size=2000)
dqn_train()
