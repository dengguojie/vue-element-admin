#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
test_BatchToSpaceNDD_impl
"""

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("BatchToSpaceNdD", None, None)

#NHWC-3D-branch_1
case1 = {
    "params": [{
        "shape": (16, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (16, 2, 32),
        "ori_format": "NHWC"
    }, {
        "shape": (8, 2, 1, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (8, 2, 32),
        "ori_format": "NHWC"
    }, [2], [1, 1]],
    "case_name": "batch_to_space_nd_d_1",
    "expect": "success",
    "support_expect": True
}

#NHWC-4D-brach_2
case2 = {
    "params": [{
        "shape": (16, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (16, 2, 2, 32),
        "ori_format": "NHWC"
    }, {
        "shape": (4, 2, 2, 2, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
        "ori_shape": (4, 2, 2, 32),
        "ori_format": "NHWC"
    }, [2, 2], [[1, 1], [1, 1]]],
    "case_name": "batch_to_space_nd_d_2",
    "expect": "success",
    "support_expect": True
}


def check_supported_1(test_arg):
    from impl.batch_to_space_nd_d import check_supported
    check_supported(
        {
            "shape": (1, 1, 1, 584, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 584, 16),
            "ori_format": "NHWC"
        }, {
            "shape": (1, 1, 1, 584, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 584, 16),
            "ori_format": "NHWC"
        }, [1], [0, 0])


def check_supported_2(test_arg):
    from impl.batch_to_space_nd_d import check_supported
    check_supported(
        {
            "shape": (1, 1, 33, 33, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 33, 33, 16),
            "ori_format": "NHWC"
        }, {
            "shape": (1, 1, 1, 33, 33, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (1, 33, 33, 16),
            "ori_format": "NHWC"
        }, [1, 1], [0, 0, 0, 0])


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_cust_test_func(test_func=check_supported_1)
ut_case.add_cust_test_func(test_func=check_supported_2)


# pylint: disable=invalid-name, unused-argument
def calc_expect_func_5hd(x, y, block_shape, crops):
    """
    calc_expect_func_5hd
    """
    shape = x['shape']
    inputArr = x['value']
    ori_format = x['ori_format']
    if ori_format == 'NCHW':
        block_shape = [block_shape[1], block_shape[2]]
        crops = [[crops[1][0], crops[1][1]], [crops[2][0], crops[2][1]]]
    batch, channel1, height, width, channel0 = shape
    block_height = height * block_shape[0]
    block_width = width * block_shape[1]
    tmp1 = inputArr.reshape(
        [block_shape[0], block_shape[1], batch // block_shape[0] // block_shape[1], channel1, height, width, channel0])
    tmp2 = tmp1.transpose(2, 3, 4, 0, 5, 1, 6)
    tmp3 = tmp2.reshape([batch // block_shape[0] // block_shape[1], channel1, block_height, block_width, channel0])
    tmp4 = tmp3[:, :, crops[0][0]:(block_height - crops[0][1]), :, :]
    outputArr = tmp4[:, :, :, crops[1][0]:(block_width - crops[1][1]), :]
    return outputArr


# pylint: disable=invalid-name, unused-argument
def calc_expect_func_6hd(x, y, block_shape, crops):
    """
    calc_expect_func_6hd
    """
    shape = x['shape']
    inputArr = x['value']
    ori_format = x['ori_format']
    if ori_format == 'NCDHW':
        block_shape = [block_shape[1], block_shape[2], block_shape[3]]
        crops = [[crops[1][0], crops[1][1]], [crops[2][0], crops[2][1]], [crops[3][0], crops[3][1]]]
    batch, depth, channel1, height, width, channel0 = shape
    block_depth = depth * block_shape[0]
    block_height = height * block_shape[1]
    block_width = width * block_shape[2]
    tmp1 = inputArr.reshape([
        block_shape[0], block_shape[1], block_shape[2], batch // block_shape[0] // block_shape[1] // block_shape[2],
        depth, channel1, height, width, channel0
    ])
    tmp2 = tmp1.transpose(3, 4, 0, 5, 6, 1, 7, 2, 8)
    tmp3 = tmp2.reshape([
        batch // block_shape[0] // block_shape[1] // block_shape[2], block_depth, channel1, block_height, block_width,
        channel0
    ])
    tmp4 = tmp3[:, crops[0][0]:(block_depth - crops[0][1]), :, :, :, :]
    tmp5 = tmp4[:, :, :, crops[1][0]:(block_height - crops[1][1]), :, :]
    outputArr = tmp5[:, :, :, :, crops[2][0]:(block_width - crops[2][1]), :]
    return outputArr


#NHWC-4D-brach_1
ut_case.add_precision_case(
    "all", {
        "params": [{
            "shape": (16, 2, 2, 2, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (16, 2, 2, 32),
            "ori_format": "NHWC",
            "param_type": "input",
            "value_range": [-10, 10]
        }, {
            "shape": (4, 2, 2, 2, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (4, 2, 2, 32),
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
            "shape": (16, 2, 2, 2, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 32, 2, 2),
            "ori_format": "NCHW",
            "param_type": "input",
            "value_range": [-10, 10]
        }, {
            "shape": (4, 2, 2, 2, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (4, 32, 2, 2),
            "ori_format": "NCHW",
            "param_type": "output"
        }, [1, 2, 2], [[0, 0], [1, 1], [1, 1]]],
        "calc_expect_func": calc_expect_func_5hd,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })
