import sklearn
from matplotlib import pyplot as plt
from torch import optim

from ML_training_D3QN_Bing import *

import os
import numpy as np
import pickle
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from collections import deque

import time
from datetime import datetime




def train_D3QN_Belief_to_Belief(schemes, window_size, n_neighbors, strategy_number):
    for schemes_index in range(len(schemes)):
        all_dataset_X = np.zeros((1, strategy_number))
        all_dataset_Y = np.zeros((1, strategy_number))
        original_belief = np.zeros((1, strategy_number))
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! " + schemes[schemes_index] + " No File")
            continue

        d3qn = D3QN(D3QNPolicy, optim.Adam)
        for file_name in file_list:
            print("data/trainning_data/" + schemes[schemes_index] + "/" + file_name)
            the_file = open("data/trainning_data/" + schemes[schemes_index] + "/" + file_name, "rb")
            # all_result_def_belief_all_result = pickle.load(the_file)
            # the_file.close()
            [D3QN_state_data_all_result, D3QN_reward_data_all_result, D3QN_action_data_all_result] = pickle.load(
                the_file)
            the_file.close()
            # trajectories = []
            for key in D3QN_state_data_all_result.keys():
                trajectories = []
                for n in range(len(D3QN_state_data_all_result[key])):
                    trajector = (
                        D3QN_state_data_all_result[key][n],
                        D3QN_reward_data_all_result[key][n],
                        D3QN_action_data_all_result[key][n]
                    )
                    trajectories.append(trajector)

                # Assuming d3qn is an instance of a class with the update method
                d3qn.update(trajectories)
        # print(all_dataset_X.shape[0])
        # window_size = 3
        # section_dataset_X = np.array([[all_dataset_X[i+j] for i in range(window_size)] for j in range(all_dataset_X.shape[0]-window_size)])
        # print(section_dataset_X)
        # print(section_dataset_X.shape)
        # print(all_dataset_Y.shape)
        # print(all_dataset_Y[window_size:].shape)
        # n_neighbors = 5
        # knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute')
        # model = knn.fit(section_dataset_X, all_dataset_Y[window_size:])
        # ================ above estimate each strategy separately ================

        # save trained model
        # Train D3QN agent
        os.makedirs("data/trained_D3QN_model_list", exist_ok=True)
        the_file = open("data/trained_D3QN_model_list/D3QN_trained_model_" + schemes[schemes_index] + ".pkl", "wb+")
        # pickle.dump(model_list, the_file)
        # the_file.close()
        torch.save(d3qn.policy, the_file)
        the_file.close()


def train_D3QN_Action_to_Belief(schemes, window_size, n_neighbors, strategy_number):
    for schemes_index in range(len(schemes)):
        all_dataset_X = np.zeros((1, window_size, strategy_number))
        all_dataset_Y = np.zeros((1, strategy_number))
        original_belief = np.zeros((1, strategy_number))
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! " + schemes[schemes_index] + " No File")
            continue

        # for each file
        d3qn = D3QN(D3QNPolicy, optim.Adam)
        for file_name in file_list:
            print("data/trainning_data/" + schemes[schemes_index] + "/" + file_name)
            the_file = open("data/trainning_data/" + schemes[schemes_index] + "/" + file_name, "rb")
            # all_result_def_belief_all_result = pickle.load(the_file)
            # the_file.close()
            [D3QN_state_data_all_result, D3QN_reward_data_all_result, D3QN_action_data_all_result] = pickle.load(
                the_file)
            the_file.close()
            # trajectories = []
            for key in D3QN_state_data_all_result.keys():
                trajectories = []
                for n in range(len(D3QN_state_data_all_result[key])):
                    trajector = (
                        D3QN_state_data_all_result[key][n],
                        D3QN_reward_data_all_result[key][n],
                        D3QN_action_data_all_result[key][n]
                    )
                    trajectories.append(trajector)

                # Assuming d3qn is an instance of a class with the update method
                d3qn.update(trajectories)

        os.makedirs("data/trained_D3QN_model_list", exist_ok=True)
        the_file = open("data/trained_D3QN_model_list/D3QN_trained_model_" + schemes[schemes_index] + ".pkl", "wb+")
        # pickle.dump(model_list, the_file)
        # the_file.close()
        torch.save(d3qn.policy, the_file)
        the_file.close()

    #     # concatenate data
    #     # for each simulation
    #     for key in all_result_def_obs_action_all_result.keys():
    #         window_x = np.zeros((window_size, strategy_number))
    #         # for each game
    #         for record in all_result_def_obs_action_all_result[key]:
    #             # update the window
    #             window_x = np.vstack([window_x, record])
    #             window_x = np.delete(window_x, 0, 0)
    #             all_dataset_X = np.vstack((all_dataset_X, [window_x]))
    #
    #         belief = np.array(all_result_def_belief_all_result[key])
    #
    #         # Aligning data
    #         all_dataset_Y = np.concatenate((all_dataset_Y, belief[1:]), axis=0)
    #         original_belief = np.concatenate((original_belief, belief[:-1]), axis=0)
    #         all_dataset_X = all_dataset_X[:-1]
    #
    #
    # model_list = []
    # total_R2_predict = 0
    # total_R2_no_predict = 0
    # total_MSE_predict = 0
    # total_MSE_no_predict = 0
    # for index in range(strategy_number):
    #     strate_dataset_X = all_dataset_X[:, :, index]
    #     strate_dataset_Y = all_dataset_Y[:, index]
    #     strate_origin_belief = original_belief[:, index]
    #     pd_strate_dataset_Y = pd.DataFrame(strate_dataset_Y)
    #     pd_strate_origin_belief = pd.DataFrame(strate_origin_belief)
    #
    #     X_train, X_test, y_train, y_test = train_test_split(strate_dataset_X, pd_strate_dataset_Y, test_size=0.1, random_state=1)
    #
    #     # KNN
    #     model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance', algorithm='brute', n_jobs=-1).fit(X_train, y_train)
    #     model_list.append(model)
    #
    #     y_predict = model.predict(X_test)
    #
    #     total_R2_predict += r2_score(y_test, y_predict)
    #     total_R2_no_predict += r2_score(y_test, pd_strate_origin_belief.iloc[y_test.index])
    #     total_MSE_predict += mean_squared_error(y_test, y_predict)
    #     total_MSE_no_predict += mean_squared_error(y_test, pd_strate_origin_belief.iloc[y_test.index])
    #
    # print(strate_dataset_X.shape)
    # print("\n total R2 score")
    # print(f"predict {total_R2_predict}")
    # print(f"no predict {total_R2_no_predict}")
    # print("total MSE")
    # print(f"predict {total_MSE_predict}")
    # print(f"predict {total_MSE_no_predict}")
    #
    # # save trained model
    # os.makedirs("data/trained_ML_model_list", exist_ok=True)
    # the_file = open("data/trained_ML_model_list/knn_trained_model_"+schemes[schemes_index]+".pkl", "wb+")
    # pickle.dump(model_list, the_file)
    # the_file.close()os.makedirs("data/trained_D3QN_model_list", exist_ok=True)
    #         the_file = open("data/trained_D3QN_model_list/d3qn_trained_model_" + schemes[schemes_index] + ".pkl", "wb+")
    #         # 在这里将 d3qn 模型保存为文件
    #         torch.save(d3qn.policy, the_file)
    #         the_file.close()


def array_normalization(_2d_array):
    sum_array = np.ones((len(_2d_array), strategy_number)) / strategy_number
    for index in range(len(_2d_array)):
        if sum(_2d_array[index]) == 0:
            continue
        else:
            sum_array[index] = _2d_array[index] / sum(_2d_array[index])

    # for array in _2d_array:
    #     if sum(array) == 0:
    #         sum_array = np.append(sum_array, 0)
    #     else:
    #         sum_array = np.append(sum_array, sum(array))
    return sum_array


def display_prediction_result(schemes, strategy_number):
    figure_high = 6
    figure_width = 7.5

    for schemes_index in range(len(schemes)):
        print(schemes[schemes_index])
        all_dataset_X = np.zeros((1, strategy_number))
        all_dataset_Y = np.zeros((1, strategy_number))
        path = "data/trainning_data/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! " + schemes[schemes_index] + " No File")
            continue
        file_name = file_list[0]
        the_file = open("data/trainning_data/" + schemes[schemes_index] + "/" + file_name, "rb")
        all_result_after_each_game_all_result = pickle.load(the_file)
        the_file.close()

        the_file = open("data/trained_ML_model_LIST/knn_trained_model_" + schemes[schemes_index] + ".pkl", "rb")
        regression_model_list = pickle.load(the_file)

        part_data = []
        part_data_predict = []
        iteration_index = 0
        array_index = 0
        S = array_normalization(all_result_after_each_game_all_result[iteration_index])
        strate_dataset_X = S[:, array_index]

        window_size = 5
        strate_dataset_X_paded = np.insert(strate_dataset_X, 0, np.zeros(window_size))

        strate_dataset_window_X = np.array([[strate_dataset_X_paded[i + j] for i in range(window_size)] for j in
                                            range(strate_dataset_X_paded.shape[0] - window_size)])

        part_data_predict = regression_model_list[array_index].predict(strate_dataset_window_X)
        # print(S_pred)

        for S_array in S:
            part_data.append(S_array[array_index])

        # for S_array in S_pred:
        #     part_data_predict.append(S_array[array_index])
        # print(part_data)
        # print(part_data_predict)

        plt.figure(figsize=(figure_width, figure_high))
        plt.plot(range(len(part_data)), part_data, label="Original Data")
        plt.plot(range(1, len(part_data) + 1), part_data, label="No Predict Data")
        plt.plot(range(len(part_data)), part_data_predict, label="Predict Data")
        plt.legend()
        plt.show()


def train_D3QN_predict_action(schemes, x_length, n_neighbors, strategy_number):
    for schemes_index in range(len(schemes)):

        path = "data/trainning_data_D3QN/" + schemes[schemes_index]
        file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        if len(file_list) == 0:
            print("!! " + schemes[schemes_index] + " No File")
            continue

        # for each file
        d3qn = D3QN(D3QNPolicy, optim.Adam)
        for file_name in file_list:
            print("data/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name)
            the_file = open("data/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name, "rb")

            [D3QN_state_data_all_result, D3QN_reward_data_all_result, D3QN_action_data_all_result] = pickle.load(
                the_file)
            the_file.close()
            # trajectories = []
            for key in D3QN_state_data_all_result.keys():
                trajectories = []
                for n in range(len(D3QN_state_data_all_result[key])):
                    trajector = (
                        D3QN_state_data_all_result[key][n],
                        D3QN_reward_data_all_result[key][n],
                        D3QN_action_data_all_result[key][n]
                    )
                    trajectories.append(trajector)

                # Assuming d3qn is an instance of a class with the update method
                d3qn.update(trajectories)


def train_D3QN_predict_action_vary_AAP(schemes, x_length, n_neighbors, strategy_number):
    AAP_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # chose list
    # AAP_list = [0.4]  # 11.22
    for AAP in AAP_list:
        d3qn = D3QN(D3QNPolicy, optim.Adam)  # 使用 D3QN 初始化模型
        print(f"AAP={AAP}")
        for schemes_index in range(len(schemes)):
            all_dataset_X = []
            all_dataset_Y = []
            path = "data_vary/AAP=" + str(AAP) + "/trainning_data_D3QN/" + schemes[schemes_index]
            file_list = [f for f in os.listdir(path) if not f.startswith('.')]
            if len(file_list) == 0:
                print("!! " + schemes[schemes_index] + " No File")
                continue

            # for each file
            for file_name in file_list:
                print("data_vary/AAP=" + str(AAP) + "/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name)
                the_file = open(
                    "data_vary/AAP=" + str(AAP) + "/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name,
                    "rb")
                [D3QN_state_data_all_result, D3QN_reward_data_all_result, D3QN_action_data_all_result] = pickle.load(
                    the_file)
                the_file.close()
                # trajectories = []
                for key in D3QN_state_data_all_result.keys():
                    trajectories = []
                    for n in range(len(D3QN_state_data_all_result[key])):
                        trajector = (
                            D3QN_state_data_all_result[key][n],
                            D3QN_reward_data_all_result[key][n],
                            D3QN_action_data_all_result[key][n]
                        )
                        trajectories.append(trajector)

                    # Assuming d3qn is an instance of a class with the update method
                    d3qn.update(trajectories)
                d3qn.plot_losses()


        # save trained model
            os.makedirs("data_vary/AAP=" + str(AAP) + "/trained_D3QN_model", exist_ok=True)
            the_file = open("data_vary/AAP=" + str(AAP) + "/trained_D3QN_model/trained_D3QN_model_" + schemes[
                schemes_index] + ".pkl", "wb+")
            # pickle.dump(model, the_file)
            torch.save(d3qn.policy, the_file)
            the_file.close()



def train_D3QN_predict_action_vary_LR(schemes, x_length, n_neighbors, strategy_number):
    # LR_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    LR_list = [0.00001]

    for LR in LR_list:
        d3qn = D3QN(D3QNPolicy, optim.Adam)
        print(f"LR={LR}")
        for schemes_index in range(len(schemes)):
            path = "data_vary/LR_D3QN=" + str(LR) + "/trainning_data_D3QN/" + schemes[schemes_index]
            file_list = [f for f in os.listdir(path) if not f.startswith('.')]
            if len(file_list) == 0:
                print("!! " + schemes[schemes_index] + " No File")
                continue

                # for each file
            for file_name in file_list:
                print("data_vary/LR_D3QN=" + str(LR) + "/trainning_data_D3QN/" + schemes[
                    schemes_index] + "/" + file_name)
                the_file = open("data_vary/LR_D3QN=" + str(LR) + "/trainning_data_D3QN/" + schemes[
                    schemes_index] + "/" + file_name, "rb")
                [D3QN_state_data_all_result, D3QN_reward_data_all_result, D3QN_action_data_all_result] = pickle.load(
                    the_file)
                the_file.close()
                # trajectories = []
                for key in D3QN_state_data_all_result.keys():
                    trajectories = []
                    for n in range(len(D3QN_state_data_all_result[key])):
                        trajector = (
                            D3QN_state_data_all_result[key][n],
                            D3QN_reward_data_all_result[key][n],
                            D3QN_action_data_all_result[key][n]
                        )
                        trajectories.append(trajector)

                    # Assuming d3qn is an instance of a class with the update method
                    d3qn.update(trajectories)
                d3qn.plot_losses()
            # save trained model
            os.makedirs("data_vary/LR_D3QN=" + str(LR) + "/trained_D3QN_model", exist_ok=True)
            the_file = open("data_vary/LR_D3QN=" + str(LR) + "/trained_D3QN_model/trained_D3QN_model_" + schemes[
                schemes_index] + ".pkl", "wb+")
            # pickle.dump(model, the_file)
            torch.save(d3qn.policy, the_file)
            the_file.close()


def train_D3QN_predict_action_vary_VUB(schemes, x_length, n_neighbors, strategy_number):
    VUB_list = np.array(range(1, 5 + 1)) * 2

    # VUB_list = [2, 10]
    for VUB in VUB_list:
        print(f"VUB={VUB}")
        for schemes_index in range(len(schemes)):
            all_dataset_state = []
            all_dataset_action = []
            all_dataset_reward = []
            path = "data_vary/VUB=" + str(VUB) + "/trainning_data_D3QN/" + schemes[schemes_index]
            file_list = [f for f in os.listdir(path) if not f.startswith('.')]
            if len(file_list) == 0:
                print("!! " + schemes[schemes_index] + " No File")
                continue

            # for each file
            d3qn = D3QN(D3QNPolicy, optim.Adam)  # 使用 D3QN 初始化模型
            for file_name in file_list:
                print("data_vary/VUB=" + str(VUB) + "/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name)
                the_file = open(
                    "data_vary/VUB=" + str(VUB) + "/trainning_data_D3QN/" + schemes[schemes_index] + "/" + file_name,
                    "rb")
                [D3QN_state_data_all_result, D3QN_reward_data_all_result, D3QN_action_data_all_result] = pickle.load(
                    the_file)
                the_file.close()
                # trajectories = []
                for key in D3QN_state_data_all_result.keys():
                    trajectories = []
                    for n in range(len(D3QN_state_data_all_result[key])):
                        trajector = (
                            D3QN_state_data_all_result[key][n],
                            D3QN_reward_data_all_result[key][n],
                            D3QN_action_data_all_result[key][n]
                        )
                        trajectories.append(trajector)

                    # Assuming d3qn is an instance of a class with the update method
                    d3qn.update(trajectories)

            # save trained PPO model
            os.makedirs("data_vary/VUB=" + str(VUB) + "/trained_D3QN_model", exist_ok=True)
            the_file = open(
                "data_vary/VUB=" + str(VUB) + "/trained_D3QN_model/trained_D3QN_model_" + schemes[
                    schemes_index] + ".pkl",
                "wb+")
            # pickle.dump(model, the_file)
            # the_file.close()

            torch.save(d3qn.policy, the_file)

            the_file.close()


if __name__ == '__main__':
    start = time.time()
    # schemes = ["DD-IPI", "DD-ML-IPI", "DD-Random-IPI"]
    # schemes = ["DD-IPI", "DD-PI"]
    schemes = ["DD-D3QN-PI", "DD-D3QN-IPI"]
    window_size = 5
    n_neighbors = 50  # 50
    strategy_number = 8
    # train_D3QN_Belief_to_Belief(schemes,window_size,n_neighbors, strategy_number)
    # train_D3QN_Action_to_Belief(schemes,window_size,n_neighbors, strategy_number)
    #display_prediction_result(schemes, strategy_number)
    x_length = 47

    # ML predict action
    d3qn_schemes = ["D3QN_collect_data_PI", "D3QN_collect_data_IPI"]
    d3qn_schemes1 = ["D3QN_collect_data_IPI"]

    # train_D3QN_predict_action(d3qn_schemes1, x_length, n_neighbors, strategy_number)
    # train_D3QN_predict_action_vary_AAP(d3qn_schemes1, x_length, n_neighbors, strategy_number)
    # train_D3QN_predict_action_vary_VUB(d3qn_schemes, x_length, n_neighbors, strategy_number)

    # LR训练模型
    train_D3QN_predict_action_vary_LR(d3qn_schemes, x_length, n_neighbors, strategy_number)
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    print("Project took", time.time() - start, "seconds.")