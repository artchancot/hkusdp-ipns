#!/usr/bin/env python3
# copyright (c) 2020 arthur chan & all rights reserved
# ann_2d_v2_0.py: artificial neural network for 2d simulation
# last updated: 02 mar 2020
# version: 2.0
# status: p
# (01) new calculation of mean squared error of output

# importing library
import numpy as np


# logistic sigmoid function
def logistic_sigmoid(input_arg):
    """logistic sigmoid function"""
    output_arg = 1 / (1 + (np.exp(-1 * input_arg)))
    return output_arg


# function for calculating values in hidden nodes
def calculate_hidden(
        input_array, weight_in_hidden, threshold_in_hidden):
    """function for calculating values in hidden nodes"""
    hidden_array = logistic_sigmoid(
        np.sum(input_array * weight_in_hidden, axis=1) + threshold_in_hidden)
    return hidden_array


# function for calculating output
def calculate_output(
        hidden_array, weight_in_output, threshold_in_output):
    """function for calculating output"""
    output_array = logistic_sigmoid(
        np.sum(hidden_array * weight_in_output, axis=1) + threshold_in_output)
    return output_array


# function for calculating output error
def calculate_output_error(output_array, desire_output_array):
    """function for calculating output error"""
    output_error = output_array * (1 - output_array) * (
            desire_output_array - output_array)
    return output_error


# function for calculating error in hidden layer
def calculate_hidden_error(output_error, hidden_array, weight_in_output):
    """function for calculating error in hidden layer"""
    hidden_error = hidden_array * (1 - hidden_array) * np.sum(
        weight_in_output.T * output_error, axis=1)
    return hidden_error


# function for adjusting weights in output layer
def adjust_output_weight(alpha, hidden_array, weight_in_output, output_error):
    """function for adjusting weights in output layer"""
    weight_in_output = weight_in_output + (alpha * output_error[
        np.newaxis].T.dot(hidden_array[np.newaxis]))
    return weight_in_output


# function for adjusting thresholds in output layer
def adjust_output_threshold(alpha, threshold_in_output, output_error):
    """function for adjusting thresholds in output layer"""
    threshold_in_output = threshold_in_output + (alpha * output_error)
    return threshold_in_output


# function for adjusting weights in hidden layer
def adjust_hidden_weight(beta, input_array, weight_in_hidden, hidden_error):
    """function for adjusting weights in hidden layer"""
    weight_in_hidden = weight_in_hidden + (beta * hidden_error[
        np.newaxis].T.dot(input_array[np.newaxis]))
    return weight_in_hidden


# function for adjusting thresholds in hidden layer
def adjust_hidden_threshold(beta, threshold_in_hidden, hidden_error):
    """function for adjusting thresholds in hidden layer"""
    threshold_in_hidden = threshold_in_hidden + (beta * hidden_error)
    return threshold_in_hidden


# function for calculating mean squared error of output
def calculate_ms_error(desire_output_array, output_array):
    """function for calculating mean squared error of output"""
    error = np.sum((desire_output_array - output_array) ** 2) / 3
    return error
