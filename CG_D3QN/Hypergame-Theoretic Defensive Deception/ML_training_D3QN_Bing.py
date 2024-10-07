# import os
# import pickle
# import random
#
# from collections import deque, namedtuple
#
# import random
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from flatbuffers.flexbuffers import F
# from torch.distributions import Categorical
#
#
# class D3QNPolicy(nn.Module):
#     def __init__(self, state_dim=48, action_dim=8):
#         super(D3QNPolicy, self).__init__()
#         # Policy network
#         self.policy_net = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim),
#             nn.Softmax(dim=-1),
#         )
#         # Value network
#         self.value_net = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, state):
#         q_values = self.policy_net(state)
#         state_value = self.value_net(state)
#         return q_values, state_value
#
#     def act(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0)
#         q_values, _ = self.forward(state)
#         action = torch.argmax(q_values, dim=-1).item()
#         return action
#
#     def evaluate(self, state, action):
#         q_values, state_value = self.forward(state)
#         action_q_value = q_values.gather(1, action)
#         return action_q_value, state_value
#
#     def predict_strategy_bundle(self, state):
#         # 将输入的状态转换为PyTorch张量
#         state = torch.from_numpy(state).float().unsqueeze(0)
#
#         # 通过模型获取策略网络的输出
#         probs, _ = self.forward(state)
#
#         # 获取概率最高的两个策略的索引
#         _, top_indices = torch.topk(probs, 2)
#
#         # 将索引转换为列表
#         best_bundle = top_indices.squeeze(0).numpy().tolist()
#
#         return best_bundle
#
#
# class D3QN:
#     def __init__(self, policy_class, optimizer_class, lr=0.01, gamma=0.99, clip_epsilon=0.2, epochs=10):
#         self.policy = policy_class(state_dim=48, action_dim=8)
#         self.optimizer = optimizer_class(self.policy.parameters(), lr=lr)
#         self.gamma = gamma
#         self.clip_epsilon = clip_epsilon
#         self.epochs = epochs
#         self.losses = []
#
#     def update(self, trajectories, gamma=0.99):
#         # losses = []
#         # rewardsum = []
#         for _ in range(self.epochs):
#             for i, (state, rewards, actions) in enumerate(trajectories):
#                 states = torch.Tensor(state)
#
#                 R = 0
#                 discounted_returns = []
#                 rewards_list = [rewards]  # 创建包含单个值的列表
#
#                 for r in reversed(rewards_list):
#                     R = r + gamma * R
#                     discounted_returns.insert(0, R)
#
#                 discounted_returns = torch.tensor(discounted_returns)
#
#                 # # Calculate advantages
#                 # # action, action_logprobs, values, _ = self.policy.evaluate(states, actions)
#                 # action, values, action_logprobs = self.policy.evaluate(states, actions)
#                 # advantages = returns - values.detach()
#                 action, values, action_logprobs = self.policy.evaluate(states, actions)
#                 advantages = discounted_returns - values.detach()
#
#                 # Calculate old and new action probabilities
#                 old_action, values, old_action_logprobs = self.policy.evaluate(states, action)
#                 old_probs = torch.exp(old_action_logprobs) + 1e-10
#                 new_probs = torch.exp(action_logprobs) + 1e-10
#
#                 # Calculate ratio
#                 ratio = new_probs / old_probs
#
#                 # Calculate surrogate loss
#                 surr1 = ratio * advantages
#                 surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
#                 policy_loss = -torch.min(surr1, surr2).mean()
#                 value_loss = (discounted_returns - values).pow(2).mean()
#
#                 # Update policy
#                 self.optimizer.zero_grad()
#                 loss = policy_loss + value_loss
#                 loss.backward()
#                 self.optimizer.step()
#
#                 self.losses.append(loss.item())
#     def plot_losses(self):
#         plt.plot(self.losses)
#         plt.title('D3QN Training Loss')
#         plt.xlabel('Iterations')
#         plt.ylabel('Loss')
#         plt.show()
#
#
# # Replay Buffer
# class ReplayBuffer:
#     def __init__(self, action_size, buffer_size, batch_size, seed):
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.seed = random.seed(seed)
#
#     def add(self, state, action, reward, next_state, done):
#         e = (state, action, reward, next_state, done)
#         self.memory.append(e)
#
#     def sample(self):
#         experiences = random.sample(self.memory, k=self.batch_size)
#
#         states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
#         actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long()
#         rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
#         next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
#         dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()
#
#         return (states, actions, rewards, next_states, dones)
#
#     def __len__(self):
#         return len(self.memory)


import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class D3QNPolicy(nn.Module):
    def __init__(self, state_dim=48, action_dim=8):
        super(D3QNPolicy, self).__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU()
        )
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def forward(self, state):
        features = self.policy_net(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.forward(state)

        action = q_values.argmax(dim=1).item()
        return action

    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        q_values = self.forward(state)
        action_q_values = q_values.gather(1, action)
        return action_q_values

    def predict_strategy_bundle(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.forward(state)
        _, top_indices = torch.topk(q_values, 2, dim=1)
        best_bundle = top_indices.squeeze(0).numpy().tolist()
        return best_bundle


class D3QN:
    def __init__(self, policy_class, optimizer_class, state_dim=48, action_dim=8, lr=0.001, gamma=0.99, batch_size=64,
                 buffer_size=10000, target_update_freq=10):
        self.policy = policy_class(state_dim=state_dim, action_dim=action_dim)
        self.target_net = policy_class(state_dim=state_dim, action_dim=action_dim)
        self.target_net.load_state_dict(self.policy.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer_class(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.target_update_freq = target_update_freq
        self.losses = []
        self.steps = 0

    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward = zip(*batch)
        return np.array(state), action, reward

    def update(self, trajectories):
        # Store the trajectories in the replay buffer
        for state, reward, action in trajectories:
            self.push(state, action, reward)

        # Perform the update if there are enough samples in the buffer
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards = self.sample()

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)

        # Compute Q-values
        q_values = self.policy(states).gather(1, actions)

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            next_actions = self.policy(states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(states).gather(1, next_actions)
            expected_q_values = rewards + (self.gamma * next_q_values)

        # Compute the loss
        loss = F.mse_loss(q_values, expected_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

        # Update the target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy.state_dict())

        self.steps += 1

    def plot_losses(self):
        import matplotlib.pyplot as plt
        plt.plot(self.losses)
        plt.title('D3QN Losses')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.show()
