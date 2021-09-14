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

def test_conv2d_data_rm_4(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 112], name='mad1', dtype="float16", attrs={'remove_pad_M': 100, 'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_5(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 90], name='mad1', dtype="float16",
                                       attrs={'remove_pad_M': 100, 'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_6(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 90], name='mad1', dtype="float16",
                                       attrs={'remove_pad_M': 100, 'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_7(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 90], name='mad1', dtype="float16",
                                       attrs={'remove_pad_M': 100, 'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_6(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 112, 16], name='mad1', dtype="float16", attrs={'remove_pad_M': 100,'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_7(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 90, 16], name='mad1', dtype="float16", attrs={'remove_pad_M': 100,'ori_format': 'NCHW'})
        conv2d_data_rm_compute(input_tensor)
    except RuntimeError:
        pass

def test_conv2d_data_rm_8(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 112, 16], name='tmp', dtype="float16", attrs={'remove_pad_M': 100,'ori_format': 'NCHW'})
        input_tensor_0 = tvm.compute(input_tensor.shape,
                                lambda n, c1, hw, c0:
                                input_tensor(n, c1, hw, c0),
                                name="input_tensor_0",
                                attrs=input_tensor.op.attrs)
        conv2d_data_rm_compute(input_tensor_0)
    except RuntimeError:
        pass

def test_conv2d_data_rm_9(test_arg):
    try:
        input_tensor = tvm.placeholder([1, 3, 90], name='tmp', dtype="float16",
                                       attrs={'remove_pad_M': 100, 'ori_format': 'NCHW'})
        input_tensor_0 = tvm.compute(input_tensor.shape,
                                lambda n, c, hw:
                                input_tensor(n, c, hw),
                                name="input_tensor_0",
                                attrs=input_tensor.op.attrs)
        conv2d_data_rm_compute(input_tensor_0)
    except RuntimeError:
        pass


ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_1)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_2)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_3)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_4)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_5)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_6)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_7)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_8)
ut_case.add_cust_test_func(test_func=test_conv2d_data_rm_9)
