# import os
# from turtle import done
#
# import gym
# import schemes
# import sklearn
# import torch
# import torch.nn as nn
# from datashape import Categorical
# from sklearn import tree
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import train_test_split
# import torch.optim as optim
# import pickle
# import random
# import numpy as np
# from collections import deque
#
#
#
# # 定义DQNPolicy网络
# # 定义Q网络
# class QNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)
#
#
# # 定义经验回放缓冲区
# class ReplayBuffer:
#     def __init__(self, buffer_size):
#         self.buffer = deque(maxlen=buffer_size)
#
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)
#
#     def __len__(self):
#         return len(self.buffer)
#
#
# class D3QN:
#     def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, lr, tau):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.tau = tau
#
#         self.primary_network = QNetwork(state_size, action_size)
#         self.target_network = QNetwork(state_size, action_size)
#         self.optimizer = optim.Adam(self.primary_network.parameters(), lr=lr)
#
#         self.replay_buffer = ReplayBuffer(buffer_size)
#
#         # 初始化目标网络权重
#         self.update_target_network(tau=1.0)
#
#     def update_target_network(self, tau=None):
#         if tau is None:
#             tau = self.tau
#
#         for target_param, local_param in zip(self.target_network.parameters(), self.primary_network.parameters()):
#             target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
#
#     def act(self, state, epsilon=0.0):
#         if random.random() > epsilon:
#             state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#             with torch.no_grad():
#                 action_values = self.primary_network(state).cpu().data.numpy()
#             return np.argmax(action_values)
#         else:
#             return random.choice(np.arange(self.action_size))
#
#     def learn(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return
#
#         experiences = self.replay_buffer.sample(self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*experiences)
#
#         states = torch.tensor(states, dtype=torch.float32)
#         actions = torch.tensor(actions, dtype=torch.int64)
#         rewards = torch.tensor(rewards, dtype=torch.float32)
#         next_states = torch.tensor(next_states, dtype=torch.float32)
#         dones = torch.tensor(dones, dtype=torch.float32)
#
#         # 当前Q值
#         q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
#         q_targets = rewards.unsqueeze(1) + (self.gamma * q_targets_next * (1 - dones.unsqueeze(1)))
#
#         q_expected = self.primary_network(states).gather(1, actions.unsqueeze(1))
#
#         loss = nn.MSELoss()(q_expected, q_targets)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # 软更新目标网络
#         self.update_target_network()
#
#
# # 训练D3QN模型
# def train_d3qn(env, d3qn_agent, n_episodes, max_t, epsilon_start, epsilon_end, epsilon_decay):
#     epsilon = epsilon_start
#     for i_episode in range(1, n_episodes + 1):
#         state = env.reset()
#         total_reward = 0
#         for t in range(max_t):
#             action = d3qn_agent.act(state, epsilon)
#             next_state, reward, done, _ = env.step(action)
#
#             d3qn_agent.replay_buffer.push(state, action, reward, next_state, done)
#             state = next_state
#
#             d3qn_agent.learn()
#
#             total_reward += reward
#             if done:
#                 break
#
#         epsilon = max(epsilon_end, epsilon_decay * epsilon)
#         print(f"Episode {i_episode}/{n_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")
#
#     return d3qn_agent