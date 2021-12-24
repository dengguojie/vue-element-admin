#!/usr/bin/env python 
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from te import tvm
from impl.util.util_conv2d_dynamic import Conv2dParaProcess

ut_case = OpUT("Conv2D", "impl.dynamic.conv2d", "conv2d")

def gen_trans_data_case(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x, expect):
    return {"params": [inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x],
            "case_name": "dynamic_conv2d_case",
            "expect": expect
            }

print("adding Conv2D dyanmic op testcases")
for test_case  in tc.conv2D_dynamic_ut_testcase:
    ut_case.add_case(test_case[0], gen_trans_data_case(*test_case[1:]))


def test_conv2d_param_process(test_arg):
    fmap = tvm.placeholder((-1, 2, 8, 8, 16), name="fmap", dtype="float16", attrs={"ori_shape": (-1, 32, 8, 8), "format": "NCHW", "ori_format": "NCHW", "range": [(1, 2), (32, 32), (8, 8), (8, 8)]})
    weight = tvm.placeholder((8, 2, 16, 16), name="weight", dtype="float16", attrs={"ori_shape": (32, 32, 2, 2), "format": "FRACTAL_Z", "ori_format": "NCHW"})
    bias_tensor = None
    offset_w_tensor = None
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]
    outputs = {}

    ori_paras = {
        "inputs": fmap, "weights": weight, "bias": bias_tensor, "offset_w": offset_w_tensor,
        "outputs": outputs, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": 1, "data_format": "float16", "kernel_name": "conv2d",
    }
    Conv2dParaProcess(ori_paras)

print("adding Connv2D dyanmic op param process")
ut_case.add_cust_test_func(test_func=test_conv2d_param_process)


def test_conv2d_param_process_dynamic():
    fmap = tvm.placeholder((-1, 2, 8, 8, 16), name="fmap", dtype="float16", attrs={"ori_shape": (-1, 32, 8, 8), "format": "NCHW", "ori_format": "NCHW", "range": [(1, 2), (32, 32), (8, 8), (8, 8)]})
    weight = tvm.placeholder((8, 2, 16, 16), name="weight", dtype="float16", attrs={"ori_shape": (32, 32, 2, 2), "format": "FRACTAL_Z", "ori_format": "NCHW"})
    bias_tensor = None
    offset_w_tensor = None
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]
    outputs = {'shape':(2, 2, 7, 7, 16), 'ori_shape':(32, 32, 7, 7), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'float16'}

    ori_paras = {
        "inputs": fmap, "weights": weight, "bias": bias_tensor, "offset_w": offset_w_tensor,
        "outputs": outputs, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": 1, "data_format": "NCHW", "kernel_name": "conv2d","dtype": "float16"
    }
    print(ori_paras)
    proc = Conv2dParaProcess(ori_paras)
    proc.check_range_valid([-1, 32, 8, 8], [(1, 2), (32, 32), (8, 8), (8, 8)], "test", "float16")

if __name__ == '__main__':
    test_conv2d_param_process_dynamic()
    ut_case.run("Ascend910A")
