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
case3 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NDHWC"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NDHWC"
    }, [1, 2, 2], [[0, 0], [1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_3",
    "expect": "success",
    "support_expect": True
}
case4 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NDHWC"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NDHWC"
    }, [1, 1, 1], [[0, 0], [0, 0], [0, 0]]],
    "case_name": "space_to_batch_nd_d_4",
    "expect": "success",
    "support_expect": True
}
case5 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NDHWC"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NDHWC"
    }, [1, 1, 1], [0, 0, 0, 0, 0, 0]],
    "case_name": "space_to_batch_nd_d_5",
    "expect": "success",
    "support_expect": True
}
case6 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]],
    "case_name": "space_to_batch_nd_d_6",
    "expect": "success",
    "support_expect": True
}
case7 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 1, 1, 1], [[0, 0], [0, 0], [1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_7",
    "expect": "success",
    "support_expect": True
}
case8 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "ND"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "ND"
    }, [1], [1]],
    "case_name": "space_to_batch_nd_d_8",
    "expect": RuntimeError,
    "support_expect": True
}
case9 = {
    "params": [{
        "shape": (8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "ND"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "ND"
    }, [1], [1]],
    "case_name": "space_to_batch_nd_d_9",
    "expect": RuntimeError,
    "support_expect": True
}
case10 = {
    "params": [{
        "shape": (8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NDHWC"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NDHWC"
    }, [1], [1]],
    "case_name": "space_to_batch_nd_d_10",
    "expect": RuntimeError,
    "support_expect": True
}
case11 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [2, 1, 1, 1], [[0, 0], [0, 0], [1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_11",
    "expect": RuntimeError,
    "support_expect": True
}
case12 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, -1, 1, 1], [[0, 0], [0, 0], [1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_12",
    "expect": RuntimeError,
    "support_expect": True
}
case13 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 1, 1, 1], [[0, 0], [0, 0], [-1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_13",
    "expect": RuntimeError,
    "support_expect": True
}
case14 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 3, 1, 1], [[0, 0], [0, 0], [1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_14",
    "expect": RuntimeError,
    "support_expect": True
}
case15 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 1, 3, 1], [[0, 0], [0, 0], [1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_15",
    "expect": RuntimeError,
    "support_expect": True
}
case16 = {
    "params": [{
        "shape": (2, 8, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 1, 3, 1], [[0, 0], [0, 0], [1, 1], [1, 1]]],
    "case_name": "space_to_batch_nd_d_16",
    "expect": RuntimeError,
    "support_expect": True
}
case17 = {
    "params": [{
        "shape": (8, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "ND"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "ND"
    }, [2], [[1, 1]]],
    "case_name": "space_to_batch_nd_d_17",
    "expect": RuntimeError,
    "support_expect": True
}
case18 = {
    "params": [{
        "shape": (2, 1, 2, 16),
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
    }, [2, 2], [[0, 0], [0, 0]]],
    "case_name": "space_to_batch_nd_d_18",
    "expect": RuntimeError,
    "support_expect": True
}
case19 = {
    "params": [{
        "shape": (4, 2, 1, 2, 16),
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
    }, [], [[0, 0], [0, 0]]],
    "case_name": "space_to_batch_nd_d_19",
    "expect": RuntimeError,
    "support_expect": True
}
case20 = {
    "params": [{
        "shape": (4, 2, 1, 2, 16),
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
    }, [1], [[0], [0, 0]]],
    "case_name": "space_to_batch_nd_d_20",
    "expect": RuntimeError,
    "support_expect": True
}
case21 = {
    "params": [{
        "shape": (4, 2, 1, 2, 16),
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
    }, [-1], [[0, 0], [0, 0]]],
    "case_name": "space_to_batch_nd_d_21",
    "expect": RuntimeError,
    "support_expect": True
}
case22 = {
    "params": [{
        "shape": (4, 2, 1, 2, 16),
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
    }, [1], [[-1, 0], [0, 0]]],
    "case_name": "space_to_batch_nd_d_22",
    "expect": RuntimeError,
    "support_expect": True
}
case23 = {
    "params": [{
        "shape": (32, 32, 32, 30, 32, 640),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 1, 1], [[8, 8], [8, 8], [8, 8]]],
    "case_name": "space_to_batch_nd_d_23",
    "expect": "success",
    "support_expect": True
}
case24 = {
    "params": [{
        "shape": (32, 32, 32, 30, 32, 1024),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NCDHW"
    }, {
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NDC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NCDHW"
    }, [1, 1, 1], [[64, 64], [64, 64], [64, 64]]],
    "case_name": "space_to_batch_nd_d_24",
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


def get_op_support_info_001(test_arg):
    from impl.space_to_batch_nd_d import get_op_support_info
    get_op_support_info(None, None, None, None, "space_to_batch_nd_d")


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case14)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case15)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case16)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case17)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case18)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case19)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case20)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case21)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case22)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case23)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case24)
ut_case.add_cust_test_func(test_func=check_supported_1)
ut_case.add_cust_test_func(test_func=check_supported_2)
ut_case.add_cust_test_func(test_func=get_op_support_info_001)


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

def test_op_select_format_001(test_arg):
    """
    test_op_select_format_001
    """
    from impl.space_to_batch_nd_d import op_select_format
    op_select_format(
        {
            "shape": (16, 1, 1, 16, 16, 16),
            "dtype": "float16",
            "format": "NDC1HWC0",
            "ori_shape": (16, 1, 1, 16, 16, 16),
            "ori_format": "NDHWC"
        }, None, [1, 2, 2], [[0, 0], [1, 1], [1, 1]])


ut_case.add_cust_test_func(test_func=test_op_select_format_001)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
