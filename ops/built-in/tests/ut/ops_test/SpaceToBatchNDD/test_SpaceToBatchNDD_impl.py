#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
test_SpaceToBatchNDD_impl
"""

import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("SpaceToBatchNdD", None, None)

#NHWC-3D-branch_1
case1 = {
    "params": [{
        "shape": (8, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NHWC"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NHWC"
    }, [2], [[1, 1]]],
    "case_name": "space_to_batch_nd_d_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [{
        "shape": (8, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NHWC"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NHWC"
    }, [2], [1, 1]],
    "case_name": "space_to_batch_nd_d_2",
    "expect": "success",
    "support_expect": True
}


def check_supported_1(test_arg):
    from impl.space_to_batch_nd_d import check_supported
    check_supported(
        {
            "shape": (1, 1, 1, 1168, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 1168, 16),
            "ori_format": "NHWC"
        }, {
            "shape": (1, 1, 1, 1168, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 1168, 16),
            "ori_format": "NHWC"
        }, [1], [0, 0])


def check_supported_2(test_arg):
    from impl.space_to_batch_nd_d import check_supported
    check_supported(
        {
            "shape": (1, 1, 66, 66, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 66, 66, 16),
            "ori_format": "NHWC"
        }, {
            "shape": (1, 1, 1, 66, 66, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 66, 66, 16),
            "ori_format": "NHWC"
        }, [1, 1], [0, 0, 0, 0])


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_cust_test_func(test_func=check_supported_1)
ut_case.add_cust_test_func(test_func=check_supported_2)


# pylint: disable=invalid-name, unused-argument
def calc_expect_func_5hd(x, y, block_shape, paddings):
    """
    calc_expect_func_5hd
    """
    shape = x['shape']
    inputArr = x['value']
    ori_format = x['ori_format']
    if ori_format == 'NCHW':
        block_shape = [block_shape[1], block_shape[2]]
        paddings = [[paddings[1][0], paddings[1][1]], [paddings[2][0], paddings[2][1]]]
    batch, channel1, height, width, channel0 = shape
    padded_height = height + paddings[0][0] + paddings[0][1]
    padded_width = width + paddings[1][0] + paddings[1][1]
    output_height = padded_height // block_shape[0]
    output_width = padded_width // block_shape[1]
    padded_data = np.pad(inputArr,
                         ((0, 0), (0, 0), (paddings[0][0], paddings[0][1]), (paddings[1][0], paddings[1][1]), (0, 0)),
                         'constant')
    tmp1 = padded_data.reshape([batch, channel1, output_height, block_shape[0], output_width, block_shape[1], channel0])
    tmp2 = tmp1.transpose((3, 5, 0, 1, 2, 4, 6))
    outputArr = tmp2.reshape([batch * block_shape[0] * block_shape[1], channel1, output_height, output_width, channel0])
    return outputArr


# pylint: disable=invalid-name, unused-argument
def calc_expect_func_6hd(x, y, block_shape, paddings):
    """
    calc_expect_func_6hd
    """
    shape = x['shape']
    inputArr = x['value']
    ori_format = x['ori_format']
    if ori_format == 'NCDHW':
        block_shape = [block_shape[1], block_shape[2], block_shape[3]]
        paddings = [[paddings[1][0], paddings[1][1]], [paddings[2][0], paddings[2][1]],
                    [paddings[3][0], paddings[3][1]]]
    batch, depth, channel1, height, width, channel0 = shape
    padded_depth = depth + paddings[0][0] + paddings[0][1]
    padded_height = height + paddings[1][0] + paddings[1][1]
    padded_width = width + paddings[2][0] + paddings[2][1]
    output_depth = padded_depth // block_shape[0]
    output_height = padded_height // block_shape[1]
    output_width = padded_width // block_shape[2]
    padded_data = np.pad(inputArr, ((0, 0), (paddings[0][0], paddings[0][1]), (0, 0), (paddings[1][0], paddings[1][1]),
                                    (paddings[2][0], paddings[2][1]), (0, 0)), 'constant')
    tmp1 = padded_data.reshape([
        batch, output_depth, block_shape[0], channel1, output_height, block_shape[1], output_width, block_shape[2],
        channel0
    ])
    tmp2 = tmp1.transpose((2, 5, 7, 0, 1, 3, 4, 6, 8))
    outputArr = tmp2.reshape([
        batch * block_shape[0] * block_shape[1] * block_shape[2], output_depth, channel1, output_height, output_width,
        channel0
    ])
    return outputArr


#NHWC-4D-brach_1
ut_case.add_precision_case(
    "all", {
        "params": [{
            "shape": (4, 2, 2, 2, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (4, 2, 2, 32),
            "ori_format": "NHWC",
            "param_type": "input",
            "value_range": [-10, 10]
        }, {
            "shape": (16, 2, 2, 2, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (16, 2, 2, 32),
            "ori_format": "NHWC",
            "param_type": "output"
        }, [2, 2], [[1, 1], [1, 1]]],
        "calc_expect_func": calc_expect_func_5hd,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })
#NCHW-4D-brach_1-fp32
ut_case.add_precision_case(
    "all", {
        "params": [{
            "shape": (4, 2, 2, 2, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (4, 32, 2, 2),
            "ori_format": "NCHW",
            "param_type": "input",
            "value_range": [-10, 10]
        }, {
            "shape": (16, 2, 2, 2, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 32, 2, 2),
            "ori_format": "NCHW",
            "param_type": "output"
        }, [1, 2, 2], [[0, 0], [1, 1], [1, 1]]],
        "calc_expect_func": calc_expect_func_5hd,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })
