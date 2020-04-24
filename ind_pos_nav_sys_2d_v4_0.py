#!/usr/bin/env python3
# copyright (c) 2020 arthur chan & all rights reserved
# ind_pos_nav_sys_2d_v4_0.py: 2d indoor positioning and navigation system
# last updated: 03 apr 2020
# version: 4.0
# status: p
# (01) replacing pos_dir_map_2d_v2_1 by pos_dir_map_2d_v3_0
# (02) adding option to perform normalisation of input distances
# (03) setting final_threshold to constant
# (04) removing comparison of error and max_error
# (05) adding option to check if ms_error_test increases consecutively for
#      100 times

# importing libraries
import matplotlib.pyplot as plt
import numpy as np

# importing custom libraries
import ann_2d_v2_0 as ann
import pos_dir_map_2d_v3_0 as pdm

# initialising editable variables
case_num = "6.10"
norm = True
check_mse = False
max_epoch = 5000
num_of_hidden = 13
alpha = 0.4
beta = 0.4

# initialising variables
epoch = 0
max_error = 0.01
final_threshold = 0.707
num_of_input = 6
num_of_output = 3
csv_data = "case_" + case_num.split('.')[0] + '_' + case_num.split('.')[1] + \
           "_data.csv"
b0, b1 = 0.0, 1.0
train_plot_step = 1
test_plot_step = 1
mse_flag = False
mse_count = 100
ms_error_past = 100.0

# reading all training and testing data from csv file
rows = np.genfromtxt(csv_data, delimiter=',')
num_of_train = int(rows[0, -1])
num_of_test = int(rows[-1, -1])
noise = rows[0, -5]

# initialising arrays
weight_in_hidden = np.random.random((num_of_hidden, num_of_input)) - 0.5
weight_in_output = np.random.random((num_of_output, num_of_hidden)) - 0.5
threshold_in_hidden = np.random.random(num_of_hidden) - 0.5
threshold_in_output = np.random.random(num_of_output) - 0.5
input_array = np.zeros(num_of_input)
input_array_test = np.zeros(num_of_input)
hidden_array = np.zeros(num_of_hidden)
output_array = np.zeros(num_of_output)
hidden_error = np.zeros(num_of_hidden)
output_error = np.zeros(num_of_output)
output_error_test = np.zeros(num_of_output)
desire_output_array = np.zeros(num_of_output)
desire_output_array_test = np.zeros(num_of_output)
max_dis = np.zeros(3)
min_dis = np.zeros(3)

# initialising lists
ms_error = []
ms_error_test = [100.0]
accuracy = []

# storing all training and testing data in numpy arrays
input_array_all = rows[:num_of_train, :6]
desire_output_array_all = rows[:num_of_train, 6:9]
input_array_test_all = rows[num_of_train:, :6]
desire_output_array_test_all = rows[num_of_train:, 6:9]

# normalising input distances
if norm:

    for a in np.arange(3):
        max_dis[a] = max(rows[:, a])
        min_dis[a] = min(rows[:, a])

    for i in np.arange(3):
        array = input_array_all[:, i]
        input_array_all[:, i] = (array - min_dis[i]) / (max_dis[i] -
                                                        min_dis[i])
    for j in np.arange(3):
        array = input_array_test_all[:, j]
        input_array_test_all[:, j] = (array - min_dis[j]) / (max_dis[j] -
                                                             min_dis[j])

c = mse_count
# training and self-testing loop
while epoch < max_epoch and ms_error_test[-1] >= max_error and not mse_flag:
    if epoch == 0:
        ms_error_test = []
    epoch = epoch + 1
    print("epoch =", epoch)
    error = 0.0
    error_test = 0.0

    # training
    for k in np.arange(num_of_train):
        input_array = input_array_all[k]

        desire_output_array = desire_output_array_all[k]

        # calculating values in hidden nodes
        hidden_array = ann.calculate_hidden(
            input_array, weight_in_hidden, threshold_in_hidden)

        # calculating output
        output_array = ann.calculate_output(
            hidden_array, weight_in_output, threshold_in_output)

        # calculating output error
        output_error = ann.calculate_output_error(
            output_array, desire_output_array)

        # calculating error in hidden layer
        hidden_error = ann.calculate_hidden_error(
            output_error, hidden_array, weight_in_output)

        # adjusting weights in output layer
        weight_in_output = ann.adjust_output_weight(
            alpha, hidden_array, weight_in_output, output_error)

        # adjusting thresholds in output layer
        threshold_in_output = ann.adjust_output_threshold(
            alpha, threshold_in_output, output_error)

        # adjusting weights in hidden layer
        weight_in_hidden = ann.adjust_hidden_weight(
            beta, input_array, weight_in_hidden, hidden_error)

        # adjusting thresholds in hidden layer
        threshold_in_hidden = ann.adjust_hidden_threshold(
            beta, threshold_in_hidden, hidden_error)

        # calculating mean squared error of output in training
        error += ann.calculate_ms_error(desire_output_array, output_array)

    # recording mean squared error of output in training
    ms_error.append(error / num_of_train)

    # self-testing
    for t in np.arange(num_of_test):
        input_array_test = input_array_test_all[t]

        desire_output_array_test = desire_output_array_test_all[t]

        # calculating values in hidden nodes
        hidden_array = ann.calculate_hidden(
            input_array_test, weight_in_hidden, threshold_in_hidden)

        # calculating output
        output_array = ann.calculate_output(
            hidden_array, weight_in_output, threshold_in_output)

        # calculating output error
        output_error_test = ann.calculate_output_error(
            output_array, desire_output_array_test)

        # calculating mean squared error of output in testing
        error_test += ann.calculate_ms_error(
            desire_output_array_test, output_array)

    # recording mean squared error of output in testing
    ms_error_test.append(error_test / num_of_test)

    # checking if mse increases consecutively for 10 times
    if check_mse:
        if ms_error_past - ms_error_test[-1] < 0:
            c -= 1

        else:
            c = mse_count

        ms_error_past = ms_error_test[-1]

        if c == 0:
            mse_flag = True

# printing final mean squared error of output in training and testing
print("final mse in training =", ms_error[-1])
print("final mse in testing =", ms_error_test[-1])

# plotting mean squared error against iteration with training data as inputs
print("[plotting mean squared error against epoch (training)!]")
x = np.arange(len(ms_error[::train_plot_step]))
y = ms_error[::train_plot_step]
plt.plot(x, y, 'r')
plt.title("Mean Squared Error (MSE) with Training Data As Inputs for Case "
          + case_num)
plt.xlabel("Epoch (1 Unit = " + str(train_plot_step) + " Epoch(s))")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

print("[mean squared error against epoch (training) is plotted!]")
plt.clf()

# plotting mean squared error against iteration with testing data as inputs
print("[plotting mean squared error against epoch (testing)!]")
xt = np.arange(len(ms_error_test[::test_plot_step]))
yt = ms_error_test[::test_plot_step]
plt.plot(xt, yt, 'b')
plt.title("Mean Squared Error (MSE) with Testing Data As Inputs for Case "
          + case_num)
plt.xlabel("Epoch (1 Unit = " + str(test_plot_step) + " Epoch(s))")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

print("[mean squared error against epoch (testing) is plotted!]")
plt.clf()

print("[training and self-testing completed!]")

# plotting position-direction map for exit 1
print("[plotting position-direction map for exit 1!]")
e = '1'
input_array[3:6] = np.array([b0, b0, b1])

accuracy.append(pdm.plot_final(
    False, case_num, e, input_array, weight_in_hidden, weight_in_output,
    threshold_in_hidden, threshold_in_output, noise, final_threshold, norm,
    max_dis, min_dis))

print("[position-direction map for exit 1 is plotted!]")
plt.clf()

# plotting position-direction map for exit 2
print("[plotting position-direction map for exit 2!]")
e = '2'
input_array[3:6] = np.array([b0, b1, b0])

accuracy.append(pdm.plot_final(
    False, case_num, e, input_array, weight_in_hidden, weight_in_output,
    threshold_in_hidden, threshold_in_output, noise, final_threshold, norm,
    max_dis, min_dis))

print("[position-direction map for exit 2 is plotted!]")
plt.clf()

# plotting position-direction map for exit 3
print("[plotting position-direction map for exit 3!]")
e = '3'
input_array[3:6] = np.array([b0, b1, b1])

accuracy.append(pdm.plot_final(
    False, case_num, e, input_array, weight_in_hidden, weight_in_output,
    threshold_in_hidden, threshold_in_output, noise, final_threshold, norm,
    max_dis, min_dis))

print("[position-direction map for exit 3 is plotted!]")
plt.clf()

print("[all position-direction maps are plotted!]")

# calculating and printing overall accuracy
o_accuracy = sum([i / 100 * 5103 for i in accuracy]) / 15309 * 100
print("overall accuracy =", np.around(o_accuracy, 1), '%')
