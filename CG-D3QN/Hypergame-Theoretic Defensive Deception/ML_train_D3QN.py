# import os
# import time
# from idlelib import tree
#
# import numpy as np
# import pickle
#
# import sklearn
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# from collections import deque
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# import os
#
# import gym
# import sklearn
# import state
# import torch
# import torch.nn as nn
# from sklearn import tree
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.model_selection import train_test_split
#
# from torch.distributions import Categorical
# import torch.optim as optim
# import pickle
#
#
# from ML_training_D3QN_Bing import state_dim, action_dim
#
#
# class D3QNPolicy(nn.Module):
#     def __init__(self, state_dim=48, action_dim=8, lr=0.001, gamma=0.99, batch_size=32):
#         super(D3QNPolicy, self).__init__()
#         self.q_network = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)
#         )
#         self.target_q_network = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)
#         )
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
#         self.loss_fn = nn.MSELoss()
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.memory = deque(maxlen=10000)
#         self.action_dim = action_dim
#
#     def forward(self, state):
#         return self.q_network(state)
#
#     def select_action(self, state, epsilon=0.1):
#         if np.random.rand() < epsilon:
#             return np.random.choice(range(self.action_dim))
#         else:
#             with torch.no_grad():
#                 q_values = self.q_network(torch.tensor(state).float().unsqueeze(0))
#                 return q_values.argmax(dim=1).item()
#
#     def predict_strategy_bundle(self, state, top_k=2):
#         """
#         根据输入的状态，预测出最佳的策略集合。
#         top_k: 返回最高的前 k 个策略
#         """
#         with torch.no_grad():
#             q_values = self.q_network(torch.tensor(state).float().unsqueeze(0))
#             _, top_indices = torch.topk(q_values, top_k)
#             best_bundle = top_indices.cpu().numpy().tolist()
#         return best_bundle
#
# class D3QN:
#     def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=10000, target_update_freq=100):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.target_update_freq = target_update_freq
#
#         self.policy_net = D3QNPolicy(state_dim, action_dim, lr)
#         self.target_net = D3QNPolicy(state_dim, action_dim, lr)
#         self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
#         self.memory = deque(maxlen=buffer_size)
#         self.loss_fn = torch.nn.MSELoss()
#         self.update_target_network()
#
#     def update_target_network(self):
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#
#     def act(self, state, epsilon=0.1):
#         if random.random() < epsilon:
#             return random.randint(0, self.action_dim - 1)
#         else:
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#             with torch.no_grad():
#                 return self.policy_net(state_tensor).argmax().item()
#
#     def remember(self, state, action, next_state, reward, done):
#         self.memory.append((state, action, next_state, reward, done))
#
#     def sample_batch(self):
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (torch.tensor(states, dtype=torch.float32),
#                 torch.tensor(actions, dtype=torch.int64),
#                 torch.tensor(rewards, dtype=torch.float32),
#                 torch.tensor(next_states, dtype=torch.float32),
#                 torch.tensor(dones, dtype=torch.float32))
#
#     def train_step(self):
#         if len(self.memory) < self.batch_size:
#             return
#
#         states, actions, rewards, next_states, dones = self.sample_batch()
#
#         q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         next_actions = self.policy_net(next_states).argmax(1)
#         next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
#         target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
#
#         loss = self.loss_fn(q_values, target_q_values.detach())
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#     def update(self, episodes=10):
#         for episode in range(episodes):
#             self.train_step()
#             if episode % self.target_update_freq == 0:
#                 self.update_target_network()
#             print(f"Episode {episode}: Training step completed.")
#
#
#
#
# def train_PPO_predict_action_vary_AAP(schemes, x_length, n_neighbors,  window_size,strategy_number):
#
#     AAP_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#
#     for AAP in AAP_list:
#         print(f"AAP={AAP}")
#         for schemes_index in range(len(schemes)):
#             state_dim = window_size * strategy_number
#             action_dim = strategy_number
#             d3qn_agent = D3QN(state_dim, action_dim)
#         path = "data_vary/AAP=" + str(AAP) + "/trainning_data/" + schemes[schemes_index]
#         file_list = [f for f in os.listdir(path) if not f.startswith('.')]
#         if len(file_list) == 0:
#             print("!! " + schemes[schemes_index] + " No File")
#             continue
#
#         experience_buffer = []
#
#         for file_name in file_list:
#             print("data/trainning_data/" + schemes[schemes_index] + "/" + file_name)
#             with open("data/trainning_data/" + schemes[schemes_index] + "/" + file_name, "rb") as the_file:
#                 all_result_def_belief_all_result = pickle.load(the_file)
#
#             for key in all_result_def_belief_all_result.keys():
#                 S = np.array(all_result_def_belief_all_result[key])
#                 S_with_zero_head = np.concatenate((np.zeros((window_size, strategy_number)), S), axis=0)
#
#                 for i in range(S_with_zero_head.shape[0] - window_size):
#                     state = S_with_zero_head[i:i + window_size].flatten()
#                     next_state = S_with_zero_head[i + 1:i + 1 + window_size].flatten()
#                     reward = np.random.rand()  # Replace with actual reward logic
#                     done = 1 if i == S_with_zero_head.shape[0] - window_size - 1 else 0
#                     action = d3qn_agent.act(state)  # Replace with actual action logic
#
#                     d3qn_agent.remember(state, action, next_state, reward, done)
#
#         # Train D3QN agent
#         d3qn_agent.update(episodes=10)
#
#         # Save the trained model
#         os.makedirs("data/trained_D3QN_model_list", exist_ok=True)
#         torch.save(d3qn_agent.policy_net.state_dict(), f"data/trained_D3QN_model_list/d3qn_trained_model_{schemes[schemes_index]}.pth")
#
#     return d3qn_agent
#
#
#
#
#
#
#
#
#
#
# def train_D3QN_predict_action_vary_AAP(schemes, x_length, n_neighbors, strategy_number):
#
#     AAP_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] #选出合适的list
#
#     for AAP in AAP_list:
#         print(f"AAP={AAP}")
#         for schemes_index in range(len(schemes)):
#             all_dataset_X = []
#             all_dataset_Y = []
#             path = "data_vary/AAP=" + str(AAP) + "/trainning_data_D3QN/" + schemes[schemes_index]
#             file_list = [f for f in os.listdir(path) if not f.startswith('.')]
#             if len(file_list) == 0:
#                 print("!! "+schemes[schemes_index]+" No File")
#                 continue
#
#             # for each file
#             for file_name in file_list:
#                 print("data_vary/AAP=" + str(AAP) + "/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name)
#                 the_file = open(
#                     "data_vary/AAP=" + str(AAP) + "/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name, "rb")
#                 [D3QN_x_data_all_result, D3QN_y_data_all_result] = pickle.load(the_file)
#                 the_file.close()
#
#                 for key in D3QN_x_data_all_result.keys():
#                     for dataset_x in D3QN_x_data_all_result[key]:
#                         all_dataset_X.append(dataset_x)
#                     for dataset_y in D3QN_y_data_all_result[key]:
#                         all_dataset_Y.append(dataset_y)
#
#             X_train, X_test, y_train, y_test = train_test_split(all_dataset_X, all_dataset_Y,
#                                                                 test_size=0.1, random_state=1)
#
#             # model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', algorithm='brute', n_jobs=-1).fit(X_train, y_train)
#             model = tree.DecisionTreeClassifier().fit(X_train, y_train)
#
#             y_predict = model.predict(X_test)
#             R2_predict = r2_score(y_test, y_predict)
#             print(f"R2_predict {R2_predict}")
#             MSE_predict = mean_squared_error(y_test, y_predict)
#             print(f"MSE_predict {MSE_predict}")
#
#             # save trained model
#             os.makedirs("data_vary/AAP=" + str(AAP) + "/trained_ML_model", exist_ok=True)
#             the_file = open("data_vary/AAP=" + str(AAP) + "/trained_ML_model/trained_classi_model_" + schemes[
#                 schemes_index] + ".pkl", "wb+")
#             pickle.dump(model, the_file)
#             the_file.close()
#
#
#
#
#
# # Usage
# if __name__ == '__main__':
#     start = time.time()
#     # schemes = ["DD-IPI", "DD-ML-IPI", "DD-Random-IPI"]
#     # schemes = ["DD-IPI", "DD-PI"]
#     # schemes = ["DD-D3QN-PI", "DD-D3QN-IPI"]
#     window_size = 5
#     n_neighbors = 50 #50
#     strategy_number = 8
#     # train_D3QN_Belief_to_Belief(schemes,window_size,n_neighbors, strategy_number)
#     # train_ML_Action_to_Belief(schemes,window_size,n_neighbors, strategy_number)
#     # display_prediction_result(schemes, strategy_number)
#     x_length = 47
#
#     # ML predict action
#     d3qn_schemes = ["trained_D3QN_model_D3QN_collect_data_PI", "trained_D3QN_model_D3QN_collect_data_IPI"]
#     d3qn_schemes1 = ["trained_D3QN_model_D3QN_collect_data_IPI"]
#
#     # train_D3QN_predict_action(d3qn_schemes1, x_length, n_neighbors, strategy_number)
#     train_D3QN_predict_action_vary_AAP(d3qn_schemes1, x_length, n_neighbors, strategy_number)
#     # train_D3QN_predict_action_vary_VUB(d3qn_schemes, x_length, n_neighbors, strategy_number)
#
#     # LR训练模型
#     # train_D3QN_predict_action_vary_LR(d3qn_schemes, x_length, n_neighbors, strategy_number)
#     print('The scikit-learn version is {}.'.format(sklearn.__version__))
#     print("Project took", time.time() - start, "seconds.")
