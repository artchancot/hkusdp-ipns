#!/usr/bin/env python3
# copyright (c) 2020 arthur chan & all rights reserved
# ind_pos_nav_sys_3d_v1_0.py: 3d indoor positioning and navigation system
# last updated: 19 apr 2020
# version: 1.0
# status: p
# (01) first release

# importing libraries
from bluepy.btle import Scanner, DefaultDelegate
from datetime import datetime
import RPi.GPIO as GPIO
import numpy as np
import Adafruit_LSM303
import math
import time
import csv
import os

# importing custom library
import ann_2d_v2_0 as ann

# initialising editable variables
norm = True
max_epoch = 5000
num_of_hidden = 13
alpha = 0.4
beta = 0.4
num_of_train = 81
num_of_test = 21

# initialising variables
epoch = 0
max_error = 0.01
final_threshold = 0.707
num_of_input = 6
num_of_output = 3
b0, b1 = 0.0, 1.0
ms_error_test = 100.0
rssi_1m = -69

# initialising variables for flow control
state = 0
count = 1
index = 0
new = True
record = False
config_compass_flag = False

# initialising list holding ble address
ble_address = ["18:93:d7:32:c3:51", "34:15:13:f7:bd:03",
               "c8:fd:19:99:86:34", "5c:31:3e:3d:25:1a",
               "6c:ec:eb:58:a4:be", "5c:f8:21:94:fd:7e"]

# generating csv file names
now = datetime.now()
timestamp = datetime.timestamp(now)
dt_object = datetime.fromtimestamp(timestamp)
dt_string = str(dt_object)
n = dt_string.find('.')
dt_substring = dt_string[0:n]
dt_ss = dt_substring.replace(':', '-')
dt_ss = dt_ss.replace(' ', '-')
csv_t_data = "ipns_t_data_" + dt_ss + ".csv"

# setting up gpio
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# initialising the lsm303 module
lsm303 = Adafruit_LSM303.LSM303()

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
compass_array = np.zeros(8)
compass_bound_array = np.zeros((2, 8))
input_array_config = np.zeros(num_of_input + num_of_output + 3)


# class for scanning nearby access points via ble
class ScanDelegate(DefaultDelegate):
    """class for scanning nearby access points via ble"""
    def __init__(self):
        DefaultDelegate.__init__(self)


# function for finding rssi
def find_rssi():
    """function for finding rssi"""
    ap1_r = 0
    ap1_l = 0
    ap2_r = 0
    ap2_l = 0
    ap3_r = 0
    ap3_l = 0

    # scanning nearby access points via ble
    scanner = Scanner().withDelegate(ScanDelegate())
    devices = scanner.scan(0.5)

    for dev in devices:
        if dev.addr == ble_address[0]:
            ap1_r = dev.rssi
        if dev.addr == ble_address[1]:
            ap1_l = dev.rssi
        if dev.addr == ble_address[2]:
            ap2_r = dev.rssi
        if dev.addr == ble_address[3]:
            ap2_l = dev.rssi
        if dev.addr == ble_address[4]:
            ap3_r = dev.rssi
        if dev.addr == ble_address[5]:
            ap3_l = dev.rssi

    ap1_rssi = ap1_r + ap1_l
    ap2_rssi = ap2_r + ap2_l
    ap3_rssi = ap3_r + ap3_l

    if ap1_r != 0 and ap1_l != 0:
        ap1_rssi = ap1_rssi / 2
    if ap2_r != 0 and ap2_l != 0:
        ap2_rssi = ap2_rssi / 2
    if ap3_r != 0 and ap3_l != 0:
        ap3_rssi = ap3_rssi / 2

    return ap1_rssi, ap2_rssi, ap3_rssi


# function for converting rssi to distance
def convert_rssi_to_distance(rssi):
    """function for converting rssi to distance"""
    dis = np.around(10**((rssi_1m - rssi) / (10 * 2)), 2)

    return dis


# function for setting training or testing input array
def set_train_test_data(cnt, in_array_config, dc):
    """function for setting training or testing input array"""

    ap1_rssi, ap2_rssi, ap3_rssi = find_rssi()

    in_array_config[0] = convert_rssi_to_distance(ap1_rssi)
    in_array_config[1] = convert_rssi_to_distance(ap2_rssi)
    in_array_config[2] = convert_rssi_to_distance(ap3_rssi)

    if cnt % 3 == 1:
        in_array_config[3:6] = [b0, b0, b1]
    elif cnt % 3 == 2:
        in_array_config[3:6] = [b0, b1, b0]
    elif cnt % 3 == 0:
        in_array_config[3:6] = [b0, b1, b1]

    if dc == 0:
        in_array_config[6:9] = [b0, b0, b0]
    elif dc == 1:
        in_array_config[6:9] = [b0, b0, b1]
    elif dc == 2:
        in_array_config[6:9] = [b0, b1, b1]
    elif dc == 3:
        in_array_config[6:9] = [b0, b1, b0]
    elif dc == 4:
        in_array_config[6:9] = [b1, b1, b0]
    elif dc == 5:
        in_array_config[6:9] = [b1, b1, b1]
    elif dc == 6:
        in_array_config[6:9] = [b1, b0, b1]
    elif dc == 7:
        in_array_config[6:9] = [b1, b0, b0]

    in_array_config[9:12] = [ap1_rssi, ap2_rssi, ap3_rssi]

    return in_array_config


# function for getting compass value
def get_compass_value():
    """function for getting compass value"""
    # reading x, y, z-axis acceleration and magnetic field strength values
    acc, m = lsm303.read()
    m_x, m_z, m_y = m

    # calculating compass heading
    compass = (math.atan2(m_y, m_x) * 180) / math.pi

    # converting compass heading to 0-360
    if compass < 0:
        compass += 360

    # converting compass heading to integer
    compass_value_int = int(compass)

    return compass_value_int


# function for determining orientation
def determine_orientation(c_value, c_b_array):
    """function for determining orientation"""
    d = 0
    for d in np.arange(8):
        f = c_b_array[1, d]
        if f < c_b_array[0, d]:
            e = c_b_array[0, d] - 360
        else:
            e = c_b_array[0, d]

        if e <= c_value < f:
            break

    return d


# function for determining direction indication
def determine_dir_indication(out_array, face):
    """function for determining direction indication"""
    if np.array_equal(out_array, [b0, b0, b0]):
        ans = 0
    elif np.array_equal(out_array, [b0, b0, b1]):
        ans = 1
    elif np.array_equal(out_array, [b0, b1, b1]):
        ans = 2
    elif np.array_equal(out_array, [b0, b1, b0]):
        ans = 3
    elif np.array_equal(out_array, [b1, b1, b0]):
        ans = 4
    elif np.array_equal(out_array, [b1, b1, b1]):
        ans = 5
    elif np.array_equal(out_array, [b1, b0, b1]):
        ans = 6
    else:
        ans = 7

    if ans == face:
        di = 0
    elif face == 0:
        di = ans
    elif ans == 0:
        di = 8 - face
    elif ans > face:
        di = ans - face
    else:
        di = 8 + (ans - face)

    return di


# function for button 17
def button_17_pressed(channel):
    """function for button 17"""
    global state
    global new
    if state == 4 and not new:
        state = 5
        new = True


# function for button 22
def button_22_pressed(channel):
    """function for button 22"""
    global state
    global new
    if state == 5 and not new:
        state = 8
        new = True
    elif state == 6 and not new:
        state = 5
        new = True
    elif state == 7 and not new:
        state = 6
        new = True
    elif state == 8 and not new:
        state = 7
        new = True


# function for button 23
def button_23_pressed(channel):
    """function for button 23"""
    global state
    global new
    global d_num
    if state == 2 and not new and not record:
        d_num += 1
        d_num = d_num % 8
        new = True
    if state == 5 and not new:
        state = 6
        new = True
    elif state == 6 and not new:
        state = 7
        new = True
    elif state == 7 and not new:
        state = 8
        new = True
    elif state == 8 and not new:
        state = 5
        new = True


# function for button 27
def button_27_pressed(channel):
    """function for button 27"""
    global state
    global new
    global count
    global record
    if state == 0:
        state = 1
    elif state == 1:
        state = 2
    elif state == 2 and config_compass_flag:
        if not record:
            record = True
    elif state == 2 and not new and not record:
        record = True
    elif state == 5 or state == 6 or state == 7 and not new:
        state = 4
        new = False
    elif state == 8 and not new:
        state = 9


# detecting button interruption
GPIO.add_event_detect(17, GPIO.FALLING, callback=button_17_pressed,
                      bouncetime=300)
GPIO.add_event_detect(22, GPIO.FALLING, callback=button_22_pressed,
                      bouncetime=300)
GPIO.add_event_detect(23, GPIO.FALLING, callback=button_23_pressed,
                      bouncetime=300)
GPIO.add_event_detect(27, GPIO.FALLING, callback=button_27_pressed,
                      bouncetime=300)

# displaying welcome message
while state == 0:
    if new:
        print("Welcome! Please press #27 to start.", "\n")
        new = False
new = True

# displaying starting configuration message
while state == 1:
    if new:
        print("Please press #27 to begin configuring the system.", "\n")
        new = False
new = True

# performing system configuration
# generating compass array
config_compass_flag = True
while True:
    compass_value = get_compass_value()

    # displaying compass heading
    print(compass_value, "\n")

    # storing critical compass value in array
    if record:
        compass_array[index] = compass_value
        print("index: " + str(index) + ' = ' + str(compass_value), "\n")
        index += 1
        time.sleep(5)
        record = False

    if index == 8:
        break
print("Compass array is generated successfully:")
print(compass_array, "\n")
time.sleep(5)
config_compass_flag = False

# generating compass bound array
for i in np.arange(8):
    a = compass_array[i-1]
    if compass_array[i] < compass_array[i-1]:
        b = compass_array[i] + 360
    else:
        b = compass_array[i]
    c = a + ((b - a) / 2)
    if c > 360:
        c -= 360
    compass_bound_array[0, i] = c
    compass_bound_array[1, i-1] = c

# generating training or testing input array
d_num = 0
print("Please press #23 to change direction.", "\n")
time.sleep(3)
while state == 2 and count <= (num_of_train + num_of_test):

    if new:
        num = count % 3
        if num == 0:
            num = 3
        print("Exit: " + str(num) + " (" + str(count) + ')', "\n")

        if d_num == 0:
            print('\u2191', "\n")
            new = False
        elif d_num == 1:
            print('\u2197', "\n")
            new = False
        elif d_num == 2:
            print('\u2192', "\n")
            new = False
        elif d_num == 3:
            print('\u2198', "\n")
            new = False
        elif d_num == 4:
            print('\u2193', "\n")
            new = False
        elif d_num == 5:
            print('\u2199', "\n")
            new = False
        elif d_num == 6:
            print('\u2190', "\n")
            new = False
        elif d_num == 7:
            print('\u2196', "\n")
            new = False

    if not new and record:
        compass_value = get_compass_value()
        orient = determine_orientation(compass_value, compass_bound_array)

        if orient != 0:
            print("Incorrect orientation! Please try again. (" + str(orient)
                  + ')', "\n")

        else:
            input_array_config = set_train_test_data(
                count, input_array_config, d_num)
            if count > num_of_train:
                row_array = np.append(input_array_config, [num_of_test])
            else:
                row_array = np.append(input_array_config, [num_of_train])

            row = row_array.tolist()

            # writing to csv file
            with open(csv_t_data, 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)

            csv_file.close()

            print("Data are stored in CSV file.", "\n")
            count += 1

        new = True
        record = False

print("All training and testing data are stored in CSV file.", "\n")

state = 3

print("Training is in progress. Please wait...", "\n")

# reading all training and testing data from csv file
rows = np.genfromtxt(csv_t_data, delimiter=',')

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

# training and self-testing loop
while epoch < max_epoch and ms_error_test >= max_error:
    epoch = epoch + 1
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

    # calculating mean squared error of output in testing
    ms_error_test = error_test / num_of_test

print("Training is completed.")
time.sleep(3)

state = 4
new = False

input_array[3:6] = [b0, b0, b1]

# operation loop
while state == 4 or state == 5 or state == 6 or state == 7 or state == 8:

    time.sleep(0.1)

    # asking if destination 1
    while state == 5 and new:
        print("Destination 1?", "\n")
        input_array[3:6] = [b0, b0, b1]
        new = False

    # asking if destination 2
    while state == 6 and new:
        print("Destination 2?", "\n")
        input_array[3:6] = [b0, b1, b0]
        new = False

    # asking if destination 3
    while state == 7 and new:
        print("Destination 3?", "\n")
        input_array[3:6] = [b0, b1, b1]
        new = False

    while state == 4 and not new:
        # measuring ble rssi
        rssi_1, rssi_2, rssi_3 = find_rssi()

        # obtaining distances
        dis1 = convert_rssi_to_distance(rssi_1)
        dis2 = convert_rssi_to_distance(rssi_2)
        dis3 = convert_rssi_to_distance(rssi_3)

        input_array[0] = dis1
        input_array[1] = dis2
        input_array[2] = dis3

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
            output_array = ann.calculate_output(
                hidden_array, weight_in_output, threshold_in_output)

        # output interpreter
        output_array[output_array >= final_threshold] = b1
        output_array[output_array != b1] = b0

        # determining orientation and direction indication
        compass_value = get_compass_value()
        orient = determine_orientation(compass_value, compass_bound_array)
        direct = determine_dir_indication(output_array, orient)

        # displaying direction
        if direct == 0:
            print('\u2191', "\n")
        elif direct == 1:
            print('\u2197', "\n")
        elif direct == 2:
            print('\u2192', "\n")
        elif direct == 3:
            print('\u2198', "\n")
        elif direct == 4:
            print('\u2193', "\n")
        elif direct == 5:
            print('\u2199', "\n")
        elif direct == 6:
            print('\u2190', "\n")
        elif direct == 7:
            print('\u2196', "\n")

        message = "j1: " + str(dis1) + ", " + str(dis2) + ", " + str(dis3)
        mqtt_cmd = "mosquitto_pub -h 192.168.0.178 -m \"" + message + \
                   "\" -t position"
        os.system(mqtt_cmd)

    # asking if exit
    while state == 8 and new:
        print("Exit?", "\n")
        new = False

# displaying goodbye message
if state == 9:
    print("Goodbye! See you next time.", "\n")

    time.sleep(5)

exit()
