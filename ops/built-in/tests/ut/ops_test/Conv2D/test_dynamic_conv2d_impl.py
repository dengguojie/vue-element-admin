#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from te import tvm
from impl.util.util_conv2d_dynamic import Conv2dParaProcess

ut_case = OpUT("Conv2D", "impl.dynamic.conv2d", "conv2d")

def gen_trans_data_case(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x, expect, transdata_index):
    return {"params": [inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x],
            "case_name": "dynamic_conv2d_case_" + str(transdata_index),
            "expect": expect
            }

print("adding Conv2D dyanmic op testcases")
for index, test_case  in enumerate(tc.conv2D_dynamic_ut_testcase):
    ut_case.add_case(test_case[0], gen_trans_data_case(*test_case[1:], index))

cdim_dynamic_ut_testcases = [
    # case 0
    ["Ascend910A",
     {"ori_shape": (1, -1, 8, 8), "shape": (1, -1, 8, 8, 16), "format": "NC1HWC0", "ori_format": "NCHW", "dtype": "float16", "range": [(1, 1), (1, 32), (8, 8), (8, 8), (16, 16)],  "ori_range": [(1, 1), (1, 32), (8, 8), (8, 8)]},
   {"shape": (8, 2, 16, 16), "ori_shape": (32, 32, 2, 2), "format": "FRACTAL_Z", "ori_format": "NCHW", "dtype": "float16"},
    None, None,
    {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'},
    (-1, -1, -1, -1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success", "cdim_dynamic_case0"]
]



cache_tiling_ut_testcases = [
    # case 0
    ["Ascend910A",
    {'ori_shape': (1, 32, -1, -1), 'shape': (1, 2, -1, -1, 16),'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]},
    {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]},
    None, None,
    {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'},
    (-1, -1, -1, -1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success", "cache_tiling_case0"]
]


def gen_cache_tilingcase_params(params):
    inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x, expect, casename = params
    return {"params": [inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x],
            "case_name": casename,
            "expect": expect
            }

for test_case  in cache_tiling_ut_testcases:
    ut_case.add_case(test_case[0], gen_cache_tilingcase_params(test_case[1:]))


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
        "groups": 1, "data_format": "float16", "kernel_name": "conv2d_param_process_case_0",
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
        "groups": 1, "data_format": "NCHW", "kernel_name": "conv2d_param_process_dynamic_case_0","dtype": "float16"
    }
    proc = Conv2dParaProcess(ori_paras)
    proc.check_range_valid([-1, 32, 8, 8], [(1, 2), (32, 32), (8, 8), (8, 8)], "test", "float16")

def test_conv2d_param_process_dynamic_cdim(test_arg):
    fmap = {"ori_shape": (1, -1, 8, 8), "shape": (1, -1, 8, 8, 16), "format": "NC1HWC0", "ori_format": "NCHW", "dtype": "float16", "range": [(1, 1), (1, 32), (8, 8), (8, 8), (16, 16)],  "ori_range": [(1, 1), (1, 32), (8, 8), (8, 8)]}
    weight = {"shape": (8, 2, 16, 16), "ori_shape": (32, 32, 2, 2), "format": "FRACTAL_Z", "ori_format": "NCHW", "dtype": "float16"}
    bias_tensor = None
    offset_w_tensor = None
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]
    outputs = {'shape':(2, 2, 7, 7, 16), 'ori_shape':(32, 32, 7, 7), "format": "NC1HWC0",  'ori_format': 'NCHW', "param_type": "output", 'dtype': 'float16'}

    ori_paras = {
        "inputs": fmap, "weights": weight, "bias": bias_tensor, "offset_w": offset_w_tensor,
        "outputs": outputs, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": 1, "data_format": "NCHW", "kernel_name": "conv2d_param_process_dynamic_cdim_case_0","dtype": "float16"
    }
    proc = Conv2dParaProcess(ori_paras)
    proc.check_only_cdim_dynamic(fmap)

print("adding Conv2D cdim dyanmic op param process")
ut_case.add_cust_test_func(test_func=test_conv2d_param_process_dynamic_cdim)

if __name__ == '__main__':
    test_conv2d_param_process_dynamic_cdim()  
    ut_case.run("Ascend910A")

