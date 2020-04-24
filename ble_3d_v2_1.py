#!/usr/bin/env python3
# copyright (c) 2020 arthur chan & all rights reserved
# ble_3d_v2_1.py: ble rssi measurement in 3d
# last updated: 19 apr 2020
# version: 2.1
# status: p
# (01) adding ble addresses of access point 2 and 3
# (02) adding calculation of average distance
# (03) adding application of average filter

# importing libraries
from bluepy.btle import Scanner, DefaultDelegate
from datetime import datetime
import RPi.GPIO as GPIO
import numpy as np
import time
import csv

# initialising editable variables
detail_switch = False
mode = 2
filter_size = 10

# initialising variables
csv_flag = False
count = 0
rssi_1m = -69

# initialising lists
ble_address = ["18:93:d7:32:c3:51", "34:15:13:f7:bd:03",
               "c8:fd:19:99:86:34", "5c:31:3e:3d:25:1a",
               "6c:ec:eb:58:a4:be", "5c:f8:21:94:fd:7e"]
target = ["0.5m", "1.0m", "1.5m", "2.0m", "2.5m", "3.0m", "3.5m", "4.0m",
          "4.5m", "5.0m", "5.5m", "6.0m"]

# generating csv file names
now = datetime.now()
timestamp = datetime.timestamp(now)
dt_object = datetime.fromtimestamp(timestamp)
dt_string = str(dt_object)
n = dt_string.find('.')
dt_substring = dt_string[0:n]
dt_ss = dt_substring.replace(':', '-')
dt_ss = dt_ss.replace(' ', '-')

# setting up gpio
GPIO.setmode(GPIO.BCM)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# class for scanning nearby access points via ble
class ScanDelegate(DefaultDelegate):
    """class for scanning nearby access points via ble"""
    def __init__(self):
        DefaultDelegate.__init__(self)


# function for converting rssi to distance
def convert_rssi_to_distance(rssi):
    """function for converting rssi to distance"""
    dis = np.around(10**((rssi_1m - rssi) / (10 * 2)), 2)

    return dis


# function for button 27
def button_27_pressed(channel):
    """function for button 27"""
    global csv_flag
    global count
    if count == 0:
        csv_flag = True
        count = 100


# detecting button interruption
GPIO.add_event_detect(27, GPIO.FALLING, callback=button_27_pressed,
                      bouncetime=300)

for i in target:
    # initialising array
    lr_af_past = np.zeros(filter_size)
    while True:
        rssi_l = 0
        rssi_r = 0

        # scanning nearby access points via ble
        scanner = Scanner().withDelegate(ScanDelegate())
        devices = scanner.scan(0.4)

        if detail_switch:
            for dev in devices:
                print("device {} ({}), RSSI = {}dB".format(dev.addr,
                                                           dev.addrType,
                                                           dev.rssi))
                for (adtype, desc, value) in dev.getScanData():
                    print("{} = {}".format(desc, value))

            time.sleep(5)

        else:
            # printing rssi of access points
            for dev in devices:
                if dev.addr == ble_address[0]:
                    rssi_r = dev.rssi

                if dev.addr == ble_address[1]:
                    rssi_l = dev.rssi

            dis_r = convert_rssi_to_distance(rssi_r)
            dis_l = convert_rssi_to_distance(rssi_l)

            avg_rssi = rssi_r + rssi_l
            if rssi_r != 0 and rssi_l != 0:
                avg_rssi = avg_rssi / 2

            avg_dis = convert_rssi_to_distance(avg_rssi)

            lr_af_past = np.roll(lr_af_past, -1)
            lr_af_past[-1] = avg_dis

            # calculating average
            s = 0.0
            for j in lr_af_past:
                if j != 0:
                    s = s + j

            avg_dis_after_filter = np.around(s / np.count_nonzero(
                lr_af_past), 2)

            print("rssi(l) =", rssi_l, "dBm    |    rssi(r) =", rssi_r,
                  "dBm    |    ", i, "    (" + str(count) + ')')
            print("dis(l) =", dis_l, "m    |    dis(r) =", dis_r,
                  'm')
            print("avg-dis =", avg_dis, 'm')

            if mode == 2:
                print("avg-dis-after-filter =", avg_dis_after_filter, 'm')

            print("\n")

            if csv_flag and count != 0:
                record = ["rssi(l)", str(rssi_l), "rssi(r)", str(rssi_r),
                          "dis(l)", dis_l, "dis(r)", dis_r, "avg-dis",
                          avg_dis, "avd-dis-filter", avg_dis_after_filter, i]
                csv_name = "rssi_" + i.split('.')[0] + '_' + i.split('.')[
                    1] + '_' + str(mode) + '_' + dt_ss + ".csv"

                # writing to csv file
                with open(csv_name, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(record)

                csv_file.close()

                count -= 1
                if count == 0:
                    csv_flag = False
                    break

        time.sleep(0.1)
