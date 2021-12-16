#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_conv(test_arg):
    from tbe import tvm
    from tbe import dsl

    shape_in = (1, 2, 16, 8, 16)
    shape_w = (2, 2, 16, 16)
    pad_h = 0
    pad_w = 0
    stride_h = 1
    stride_w = 1
    filter_h = 1
    filter_w = 1
    Data = tvm.placeholder(shape_in, name="FmapW", dtype="float16")
    Weight = tvm.placeholder(shape_w, name="FilterW", dtype="float16")
    bias_tensor = tvm.placeholder(
            (shape_w[1] * shape_w[2], ), name="Bias", dtype="float16")
    res_tensor = dsl.conv(
            Data, Weight, {"bias_tensor": bias_tensor,
                           "pad_h": pad_h, "pad_w": pad_w,
                           "stride_h": stride_h, "stride_w": stride_w,
                           "filter_h": filter_h, "filter_w": filter_w,
                           "offset_a": 0})




print("adding Conv2D tbe/dsl/conv ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_conv)
