import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import textwrap
import pylab

def display_uncertainty_varying_LR_HG_CG_D3QN_DD():
    # attacker uncertainty average result
    # schemes =  ["DD-IPI", "DD-Random-IPI", "DD-PI"]#, "DD-PI", "DD-Random-PI", "DD-ML-PI"]

    for schemes_index in range(len(schemes)):

        plt.figure(figsize=(figure_width, figure_high))
        for LR_list_index in range(len(learning_rates)):
            # the_file = open("data/" + schemes[schemes_index] + "/R3/attacker_uncertainty.pkl", "rb")
            the_file = open(
                "data_vary/LR_D3QN=" + str(learning_rates[LR_list_index]) + "/" + schemes[
                    schemes_index] + "/R3/attacker_uncertainty.pkl",
                "rb")
            att_uncertainty_history = pickle.load(the_file)
            the_file.close()


            max_length = 0
            max_index = 0
            for key in att_uncertainty_history.keys():
                if max_length < len(att_uncertainty_history[key]):
                    max_length = len(att_uncertainty_history[key])
                    max_index = key

            # print(att_uncertainty_history[max_index])
            # plt.plot(range(len(att_uncertainty_history[max_index])), att_uncertainty_history[max_index])

            average_att_uncertainty = []
            for index in range(max_length):
                sum_on_index = 0
                number_on_index = 0
                for key in att_uncertainty_history.keys():
                    if len(att_uncertainty_history[key]) > 0:
                        if len(att_uncertainty_history[key][0]) > 0:
                            # sum_on_index += att_uncertainty_history[key][0]
                            sum_on_index += np.sum(att_uncertainty_history[key][0]) / len(
                                att_uncertainty_history[key][0])
                            att_uncertainty_history[key].pop(0)
                            number_on_index += 1
                average_att_uncertainty.append(sum_on_index / number_on_index)

            x_values = range(len(average_att_uncertainty))
            y_values = average_att_uncertainty
            plt.plot(x_values, y_values, linestyle=all_linestyle[LR_list_index], label=learning_rates_name[LR_list_index],
                     linewidth=figure_linewidth, marker=marker_list[LR_list_index], markevery=50,
                     markersize=marker_size)
        plt.legend(prop={"size": legend_size / 1.2}, ncol=4, bbox_to_anchor=(-0.18, 1, 1.2, 0),
                   # ncol=2, bbox_to_anchor=(0, 1, 1, 0),
                   loc='lower left', fontsize='large', mode="expand")
        # plt.legend(prop={"size":legend_size}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel("round of games", fontsize=font_size)
        plt.ylabel( " HG-CG-D3QN-DD - AU", fontsize=font_size)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.xlim([0, max_x_length])  # fix x axis range
        plt.tight_layout()
        # os.makedirs("Figure/All-In-One", exist_ok=True)
        # plt.savefig("Figure/All-In-One/att-uncertain-AllInOne.svg", dpi=figure_dpi)
        # plt.savefig("Figure/All-In-One/att-uncertain-AllInOne.png", dpi=figure_dpi)
        os.makedirs("Figure/All-In-One/varying_LR", exist_ok=True)
        plt.savefig("Figure/All-In-One/varying_LR/att-uncertain-AllInOne.svg", dpi=figure_dpi)
        plt.savefig("Figure/All-In-One/varying_LR/att-uncertain-AllInOne.eps", dpi=figure_dpi)
        plt.savefig("Figure/All-In-One/varying_LR/att-uncertain-AllInOne.png", dpi=figure_dpi)
        plt.show()

        # defender uncertainty average result

        plt.figure(figsize=(figure_width, figure_high))
        for LR_list_index in range(len(learning_rates)):
            # the_file = open("data/" + schemes[schemes_index] + "/R3/defender_uncertainty.pkl", "rb")
            the_file = open(
                "data_vary/LR_D3QN=" + str(learning_rates[LR_list_index]) + "/" + schemes[
                    schemes_index] + "/R3/defender_uncertainty.pkl",
                "rb")
            def_uncertainty_history = pickle.load(the_file)
            the_file.close()

            max_length = 0
            for key in def_uncertainty_history.keys():
                if max_length < len(def_uncertainty_history[key]):
                    max_length = len(def_uncertainty_history[key])

            average_def_uncertainty = []
            for index in range(max_length):
                sum_on_index = 0
                number_on_index = 0
                for key in def_uncertainty_history.keys():
                    if len(def_uncertainty_history[key]) > 0:
                        sum_on_index += def_uncertainty_history[key][0]
                        # sum_on_index += np.sum(def_uncertainty_history[key][0])/len(def_uncertainty_history[key][0])
                        def_uncertainty_history[key].pop(0)
                        number_on_index += 1
                average_def_uncertainty.append(sum_on_index / number_on_index)

            x_values = range(len(average_def_uncertainty))
            y_values = average_def_uncertainty
            plt.plot(x_values, y_values, linestyle=all_linestyle[LR_list_index], label=learning_rates_name[LR_list_index],
                     linewidth=figure_linewidth, marker=marker_list[LR_list_index], markevery=50,
                     markersize=marker_size)
        plt.legend(prop={"size": legend_size / 1.2}, ncol=4, bbox_to_anchor=(-0.18, 1, 1.2, 0),
                   # ncol=2, bbox_to_anchor=(0, 1, 1, 0),
                   loc='lower left', fontsize='large', mode="expand")
        plt.xlabel("round of games", fontsize=font_size)
        plt.ylabel(" HG-CG-D3QN-DD - DU", fontsize=font_size)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.xlim([0, max_x_length])  # fix x axis range
        plt.tight_layout()

        # os.makedirs("Figure/All-In-One", exist_ok=True)
        # plt.savefig("Figure/All-In-One/def-uncertain-AllInOne.svg", dpi=figure_dpi)
        # plt.savefig("Figure/All-In-One/def-uncertain-AllInOne.png", dpi=figure_dpi)
        os.makedirs("Figure/All-In-One/varying_LR", exist_ok=True)
        plt.savefig("Figure/All-In-One/varying_LR/def-uncertain-AllInOne.svg", dpi=figure_dpi)
        plt.savefig("Figure/All-In-One/varying_LR/def-uncertain-AllInOne.eps", dpi=figure_dpi)
        plt.savefig("Figure/All-In-One/varying_LR/def-uncertain-AllInOne.png", dpi=figure_dpi)
        plt.show()













if __name__ == '__main__':
    # preset values
    all_linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    patterns = ["|", "\\", "/", "+", "-", ".", "*", "x", "o", "O"]
    font_size = 20  # 25
    figure_high = 5  # 6
    figure_width = 7.5
    figure_linewidth = 3
    figure_dpi = 100
    legend_size = 15  # 18
    axis_size = 20
    marker_size = 12
    marker_list = ["p", "d", "v", "x", "s", "*", "1", ".", "6", "h"]
    strategy_number = 8
    max_x_length = 60
    use_legend = False

    # legend_name = ["HG-CG-D3QN-DD", "TG-CG-D3QN-DD", "HG-DD", "TG-DD", "RG-DD", "RG-ND"]
    # schemes = ["DD-D3QN-IPI", "DD-D3QN-PI", "DD-IPI", "DD-PI", "DD-Random", "No-DD-Random"]
#about lr-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    learning_rates_name = ["lr=0.00001", "lr=0.0001", "lr=0.001", "lr=0.01", "lr=0.1"]

    #for HG-CG-D3QN-DD
    # legend_name = ["HG-CG-D3QN-DD"]
    # schemes = ["DD-D3QN-IPI"]
    # display_uncertainty_varying_LR_HG_CG_D3QN_DD()

    # for TG-CG-D3QN-DD
    legend_name = ["TG-CG-D3QN-DD"]
    schemes = ["DD-D3QN-PI"]
    display_uncertainty_varying_LR_HG_CG_D3QN_DD()

