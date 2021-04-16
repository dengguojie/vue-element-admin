#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from te import tvm
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from impl.conv2d_data_rm import conv2d_data_rm_compute
from impl.conv2d import get_op_support_info

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")



def test_conv2d_data_rm_1(test_arg):
    try:
        input_tensor = []
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_2(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 16, 16, 16], name='input_tensor', dtype="float16", attrs={'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_3(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 16, 16], name='input_tensor', dtype="float32", attrs={'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass



ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_1)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_2)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_3)






