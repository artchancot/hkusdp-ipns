#!/usr/bin/env python3
# copyright (c) 2020 arthur chan & all rights reserved
# pos_dir_map_2d_v3_0.py: plotting 2d position-direction map
# last updated: 03 apr 2020
# version: 3.0
# status: p
# (01) adding normalisation of input distances when calculating all outputs
# (02) setting distance to 0 if distance is negative after adding noise

# importing libraries
import matplotlib.pyplot as plt
import numpy as np

# importing custom library
import ann_2d_v2_0 as ann

# initialising variables
csv_ideal = "ideal_map_data.csv"
b0, b1 = 0.0, 1.0

# initialising lists
colour = ["#e6194b", "#3cb44b", "#ffd700", "#4363d8", "#f58231", "#911eb4",
          "#42d4f4", "#f032e6"]
wall_1 = [[0.0, 2.0], [0.0, 0.0], [2.0, 2.0]]
wall_2 = [[4.0, 6.0], [0.0, 0.0], [2.0, 2.0]]
wall_3 = [[0.0, 6.0], [4.0, 4.0], [4.5, 4.5]]

# initialising arrays
ap_pos_1 = np.array([3.0, 4.0])
ap_pos_2 = np.array([2.0, 2.0])
ap_pos_3 = np.array([4.0, 2.0])


def calculate_user_ap_dis(pos_array):
    """function for calculating distances between all positions and access
    points"""
    # initialising arrays
    user_ap_dis_array = np.zeros((1701, 3))

    # calculating the distances between all positions and access points
    user_ap_dis_array[:, 0] = np.sqrt((pos_array[:, 0] - ap_pos_1[0]) ** 2 + (
            pos_array[:, 1] - ap_pos_1[1]) ** 2)
    user_ap_dis_array[:, 1] = np.sqrt((pos_array[:, 0] - ap_pos_2[0]) ** 2 + (
            pos_array[:, 1] - ap_pos_2[1]) ** 2)
    user_ap_dis_array[:, 2] = np.sqrt((pos_array[:, 0] - ap_pos_3[0]) ** 2 + (
            pos_array[:, 1] - ap_pos_3[1]) ** 2)

    return user_ap_dis_array


# function for calculating direction codes for all positions
def calculate_all_output(
        input_array, weight_in_hidden, weight_in_output,
        threshold_in_hidden, threshold_in_output, user_ap_dis_array,
        final_threshold, norm, max_dis, min_dis):
    """function for calculating direction codes for all positions"""
    output_array = np.zeros((1701, 3))

    for i in np.arange(1701):
        input_array[0:3] = user_ap_dis_array[i]

        # normalising input distances
        if norm:
            for k in np.arange(3):
                if input_array[k] >= max_dis[k]:
                    input_array[k] = max_dis[k]
                if input_array[k] <= min_dis[k]:
                    input_array[k] = min_dis[k]

                input_array[k] = (input_array[k] - min_dis[k]) / (max_dis[k]
                                                                  - min_dis[
                                                                      k])

        # calculating values in hidden nodes
        hidden_array = ann.calculate_hidden(
            input_array, weight_in_hidden, threshold_in_hidden)

        # calculating output
        output_array_tmp = ann.calculate_output(
            hidden_array, weight_in_output, threshold_in_output)

        # output interpreter
        output_array_tmp[output_array_tmp >= final_threshold] = b1
        output_array_tmp[output_array_tmp != b1] = b0

        output_array[i] = output_array_tmp

    return output_array


# function for plotting position-direction map
def plot_position_direction_map(i, j, u, v, c):
    """function for plotting position-direction map"""
    plt.quiver(i, j, u, v, color=c, angles='xy', scale_units='xy', scale=1,
               zorder=3)
    plt.axis([-0.5, 6.1 + 0.5, -0.5, 4.1 + 1.0])


# function for assigning colours to each position and plotting map
def assign_colour_and_plot_map(pos_output_array, c):
    """function for assigning colours to all positions according to their
    direction codes"""
    for i in np.arange(1701):
        if np.array_equal(pos_output_array[i, 2:5], [b0, b0, b0]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], 0, 0.1, c[0])
        elif np.array_equal(pos_output_array[i, 2:5], [b0, b0, b1]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], -0.0707,
                0.0707, c[1])
        elif np.array_equal(pos_output_array[i, 2:5], [b0, b1, b1]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], -0.1, 0, c[2])
        elif np.array_equal(pos_output_array[i, 2:5], [b0, b1, b0]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], -0.0707,
                -0.0707, c[3])
        elif np.array_equal(pos_output_array[i, 2:5], [b1, b1, b0]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], 0, -0.1, c[4])
        elif np.array_equal(pos_output_array[i, 2:5], [b1, b1, b1]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], 0.0707,
                -0.0707, c[5])
        elif np.array_equal(pos_output_array[i, 2:5], [b1, b0, b1]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], 0.1, 0, c[6])
        elif np.array_equal(pos_output_array[i, 2:5], [b1, b0, b0]):
            plot_position_direction_map(
                pos_output_array[i, 0], pos_output_array[i, 1], 0.0707,
                0.0707, c[7])


# function for drawing walls in position-direction map
def draw_wall():
    """function for drawing walls in position-direction map"""
    plt.fill_between(
        wall_1[0], wall_1[1], wall_1[2], facecolor="#808080", alpha=0.3)
    plt.fill_between(
        wall_2[0], wall_2[1], wall_2[2], facecolor="#808080", alpha=0.3)
    plt.fill_between(
        wall_3[0], wall_3[1], wall_3[2], facecolor="#808080", alpha=0.3)


# function for plotting position-direction map
def plot_final(
        i_switch, case_num, exit_num, input_array=None,
        weight_in_hidden=None, weight_in_output=None, threshold_in_hidden=None,
        threshold_in_output=None, noise=0.0, final_threshold=0.8, norm=False,
        max_dis=None, min_dis=None):
    """function for plotting position-direction map"""
    # reading all pos data from csv file
    rows_ideal = np.genfromtxt(csv_ideal, delimiter=',')
    pos_array = rows_ideal[0:1701, 0:2]

    # calculating distances between all positions and access points
    user_ap_dis_array = calculate_user_ap_dis(pos_array)

    # drawing walls in position-direction map
    draw_wall()

    # extracting ideal output data
    if exit_num == '1':
        ideal_output_array = rows_ideal[0:1701, 2:5]
    elif exit_num == '2':
        ideal_output_array = rows_ideal[1701:3402, 2:5]
    else:
        ideal_output_array = rows_ideal[3402:5103, 2:5]

    # generating output array
    if not i_switch:
        plt.title("Position-Direction Map for Exit " + exit_num + " in Case " +
                  case_num)
        plt.grid(True, zorder=0)

        # adding noise to all positions
        user_ap_dis_array += np.random.uniform(noise * -1, noise, np.shape(
            user_ap_dis_array))
        user_ap_dis_array = np.round(user_ap_dis_array, 1)
        user_ap_dis_array[user_ap_dis_array < 0] = 0.0

        # calculating direction codes for all positions
        output_array = calculate_all_output(
            input_array, weight_in_hidden, weight_in_output,
            threshold_in_hidden, threshold_in_output, user_ap_dis_array,
            final_threshold, norm, max_dis, min_dis)

        # calculating accuracy of actual output
        accuracy = np.sum(
            np.equal(ideal_output_array, output_array)) / 5103 * 100
        print("accuracy =", np.around(accuracy, 1), '%')
    else:
        plt.title("Ideal Position-Direction Map for Exit " + exit_num)
        plt.grid(True, zorder=0)

        output_array = ideal_output_array
        accuracy = 100.0

    # combining pos_array and output_array
    pos_output_array = np.append(pos_array, output_array, axis=1)

    # assigning colours to each position and plotting map
    assign_colour_and_plot_map(pos_output_array, colour)

    plt.show()

    return accuracy


# function for plotting ideal position-direction maps
def plot_ideal():
    """function for plotting ideal position-direction maps"""
    # plotting ideal position-direction map for exit 1
    print("[plotting ideal position-direction map for exit 1!]")
    plot_final(True, None, '1')

    print("[ideal position-direction map for exit 1 is plotted!]")
    plt.clf()

    # plotting ideal position-direction map for exit 2
    print("[plotting ideal position-direction map for exit 2!]")
    plot_final(True, None, '2')

    print("[ideal position-direction map for exit 2 is plotted!]")
    plt.clf()

    # plotting ideal position-direction map for exit 3
    print("[plotting ideal position-direction map for exit 3!]")
    plot_final(True, None, '3')

    print("[ideal position-direction map for exit 3 is plotted!]")
    plt.clf()
