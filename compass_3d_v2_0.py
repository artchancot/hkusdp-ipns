#!/usr/bin/env python3
# copyright (c) 2020 arthur chan & all rights reserved
# compass_3d_v2_0.py: testing of lsm303 module
# last updated: 18 apr 2020
# version: 2.0
# status: p
# (01) storing critical compass value in list

# importing libraries
import Adafruit_LSM303
import RPi.GPIO as GPIO
import time
import math

# initialising editable variable
print_all_switch = False

# initialising variables
record_compass_flag = False
index = 0

# initialising list
compass_list = [0, 0, 0, 0, 0, 0, 0, 0]

# initialising the lsm303 module
lsm303 = Adafruit_LSM303.LSM303()

# setting up gpio
GPIO.setmode(GPIO.BCM)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# function for button 27
def button_27_pressed(channel):
    """function for button 27"""
    global record_compass_flag
    if not record_compass_flag:
        record_compass_flag = True


# detecting button interruption
GPIO.add_event_detect(27, GPIO.FALLING, callback=button_27_pressed,
                      bouncetime=300)

while True:
    # reading x, y, z-axis acceleration and magnetic field strength values
    a, m = lsm303.read()
    a_x, a_y, a_z = a
    m_x, m_z, m_y = m

    # calculating compass heading
    compass_value = (math.atan2(m_y, m_x) * 180) / math.pi

    # converting compass heading to 0-360
    if compass_value < 0:
        compass_value += 360

    # converting compass heading to integer
    compass_value_int = int(compass_value)

    # displaying compass heading
    print(compass_value_int, "\n")

    # storing critical compass value in list
    if record_compass_flag:
        compass_list[index] = compass_value_int
        print("index: " + str(index) + ' = ' + str(compass_value_int))
        index += 1
        time.sleep(5)
        record_compass_flag = False

    if print_all_switch:
        # printing all readings for testing
        print("a_x={0}, a_y={1}, a_z={2}, m_x={3}, m_y={4}, m_z={5}".format(
            a_x, a_y, a_z, m_x, m_y, m_z))

    time.sleep(0.3)

    if index == 8:
        break

# printing all critical compass values
print(compass_list)
