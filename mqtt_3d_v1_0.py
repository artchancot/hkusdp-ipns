#!/usr/bin/env python3
# copyright (c) 2020 arthur chan & all rights reserved
# mqtt_3d_v1_0.py: receiving position data from user in 3d
# last updated: 18 apr 2020
# version: 1.0
# status: p
# (01) first release

# importing library
import paho.mqtt.subscribe as subscribe

# printing payload
while True:
    msg = subscribe.simple("position", hostname="192.168.0.178")
    print("%s: %s" % (msg.topic, msg.payload.decode()))
