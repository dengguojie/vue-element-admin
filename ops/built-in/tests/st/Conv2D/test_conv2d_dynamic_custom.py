#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm
from impl.util.util_conv2d_dynamic import Conv2dParaProcess
from impl.dynamic.conv2d import conv2d


def test_conv2d_param_process():
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
    proc.get_output_range([2, 1, 4096, 4096],[(1, None), (1, 1), (1, None), (1, None)])


# cache tiling ut testcases
cache_tiling_ut_testcases = [
    # case 0
    ["Ascend910A",
    {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]},
    {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW", "range": [(64, 64), (32, 32), (3, 3), (3, 3)]},
    None, None,
    {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'},
    (-1, -1, -1, -1), (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success", "cache_tiling_case0"]
]


def test_cachetiling_conv2d():
    for test_case  in cache_tiling_ut_testcases:
        inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x, expect, casename = test_case[1:]
        from tbe.common.context import op_context
        import tbe.dsl.base.operation as operation
        with op_context.OpContext():
            with operation.dynamic():
                conv2d(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x, casename)

def test_conv2d_param_process_dynamic_cdim():
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
        "groups": 1, "data_format": "NCHW", "kernel_name": "conv2d","dtype": "float16"
    }
    proc = Conv2dParaProcess(ori_paras)
    proc.check_only_cdim_dynamic(fmap)

if __name__ == '__main__':
    test_conv2d_param_process_dynamic_cdim()
    test_conv2d_param_process()
    test_cachetiling_conv2d()
