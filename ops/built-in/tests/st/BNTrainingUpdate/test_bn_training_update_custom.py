#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

from impl.dynamic.bn_training_update import _check_shape

def test_bn_training_update_check_shape1():
    input_list1 = [[32, 32, 28, 28, 16],
                   [1, 2, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   "NC1HWC0"]
    _check_shape(*input_list1)

def test_bn_training_update_check_shape2():
    input_list2 = [[32, 32, 28, 28, 16],
                   [1, 32, 1, 1, 16],
                   [1, 2, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   "NC1HWC0"]
    _check_shape(*input_list2)

def test_bn_training_update_check_shape3():
    input_list3 = [[32, 32, 28, 28, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 2, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   "NC1HWC0"]
    _check_shape(*input_list3)

def test_bn_training_update_check_shape4():
    input_list4 = [[32, 32, 28, 28, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 2, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   "NC1HWC0"]
    _check_shape(*input_list4)

def test_bn_training_update_check_shape5():
    input_list5 = [[32, 32, 28, 28, 16],
                   [1, 2, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 2, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   "NC1HWC0"]
    _check_shape(*input_list5)

def test_bn_training_update_check_shape6():
    input_list6 = [[32, 32, 28, 28, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 32, 1, 1, 16],
                   [1, 2, 1, 1, 16],
                   "NC1HWC0"]
    _check_shape(*input_list6)

if __name__ == '__main__':
    print("enter est_bn_training_update_check_shape")
    try:
        test_bn_training_update_check_shape1()
    except RuntimeError:
        print("Dimension C of x and sum must be equal")
    try:
        test_bn_training_update_check_shape2()
    except RuntimeError:
        print("Dimension C of x and sum must be equal")
    try:
        test_bn_training_update_check_shape3()
    except RuntimeError:
        print("Dimension C of x and sum must be equal")
    try:
        test_bn_training_update_check_shape4()
    except RuntimeError:
        print("Dimension C of x and sum must be equal")
    try:
        test_bn_training_update_check_shape5()
    except RuntimeError:
        print("Dimension C of x and sum must be equal")
    try:
        test_bn_training_update_check_shape6()
    except RuntimeError:
        print("Dimension C of x and sum must be equal")