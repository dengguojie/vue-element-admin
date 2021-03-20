#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseConv2DBackpropInput", "impl.dynamic.depthwise_conv2d_backprop_input",
               "depthwise_conv2d_backprop_input")
dynamic_conv2d_bp_input_op_testcase = [
    ((3, 3, 960, 1), (32, 960, 7, 7), (32, 960, 7, 7), 1, 1, (1, 1, 1, 1), "NCHW", [0], "success"),
    ((5, 5, 192, 1), (16, 192, 14, 28), (16, 192, 28, 28), 2, 1, (-1, -1, -1, -1), "NCHW", [0, 2], "success"),
    ((3, 3, 96, 1), (8, 96, 56, 56), (8, 96, 56, 56), 1, 1, (-1, -1, -1, -1), "NCHW", [0, 3], "success"),
    ((1, 1, 32, 1), (1, 32, 112, 122), (1, 32, 112, 112), 2, 1, (0, 0, 0, 0), "NCHW", [2, 3], "success"),
    ((3, 3, 256, 1), (2, 334, 502, 256), (2, 336, 504, 256), 1, 1, (0, 0, 0, 0), "NHWC", [0, 2, 3], "success"),
    ((5, 5, 240, 1), (2, 14, 14, 240), (2, 28, 28, 240), 2, 1, (-1, -1, -1, -1), "NHWC", [0, 2, 3], "success"),
]


def _get_kernel_name(x_shape, filter_shape, bias_shape, stride, dilation, pads, dtype):
    bias_shape = bias_shape if bias_shape else []
    padding = "SAME" if -1 in pads else "VALID"
    kernel_name = 'dp_conv2d_' + '_'.join(map(str, x_shape)) + '_' + '_'.join(map(str, filter_shape)) + '_' + '_'.join(
        map(str, bias_shape)) + '_' + str(stride) + '_' + str(dilation) + "_" + padding + '_' + dtype
    return kernel_name


def _shape_to_NC1HWC0(shape, data_format, dtype):
    if data_format.upper() == "NCHW":
        n, c, h, w = shape
    else:  # NCHW
        n, h, w, c = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (n, c1, h, w, c0)


def _shape_to_C1HWNCoC0(shape, data_format, dtype):
    if data_format.upper() == "HWCN":
        h, w, c, n = shape
    else:  # NCHW
        n, c, h, w = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (c1, h, w, n, c0, c0)


def _get_range_from_shape(shape, dynamic_dim=[]):
    ori_range = [(dim, dim) for dim in shape]
    if dynamic_dim:
        for dim in dynamic_dim:
            ori_range[dim] = (max(1, shape[dim] // 2), min(4096, shape[dim] * 2))
    return ori_range


def _trans_dynamic_shape(shape, format, dynamic_dim):
    shape = list(shape)
    if 0 in dynamic_dim:
        n_dim = format.index("N")
        shape[n_dim] = -1
    if 2 in dynamic_dim:
        h_dim = format.index("H")
        shape[h_dim] = -1
    if 3 in dynamic_dim:
        w_dim = format.index("W")
        shape[w_dim] = -1
    return tuple(shape)


def _gen_trans_data_case(param):
    filter_ori_shape, out_backprop_ori_shape, input_size, stride, dilation, pads, data_format, dynamic_dim, expect_result = param

    data_format = data_format.upper()
    dtype = "float16"

    filter_shape = _shape_to_C1HWNCoC0(filter_ori_shape, "HWCN", dtype),
    out_backprop_shape = _shape_to_NC1HWC0(out_backprop_ori_shape, data_format, dtype)
    input_grad_shape = _shape_to_NC1HWC0(input_size, data_format, dtype)
    x = {
        "shape": [4],
        "format": "NC1HWC0",
        "ori_shape": [4],
        "ori_format": data_format,
        "dtype": dtype,
        "range": _get_range_from_shape(input_grad_shape, dynamic_dim)
    }
    filter = {
        "shape": filter_shape,
        "ori_shape": filter_ori_shape,
        "ori_format": "HWCN",
        "format": "C1HWNCoC0",
        "dtype": dtype,
        "range": _get_range_from_shape(filter_shape)
    }
    out_backprop = {
        "shape": _trans_dynamic_shape(out_backprop_shape, "NC1HWC0", dynamic_dim),
        "format": "NC1HWC0",
        "ori_shape": _trans_dynamic_shape(out_backprop_ori_shape, data_format, dynamic_dim),
        "ori_format": data_format,
        "dtype": dtype,
        "range": _get_range_from_shape(out_backprop_shape, dynamic_dim)
    }
    input_grad = {
        "shape": _trans_dynamic_shape(input_grad_shape, "NC1HWC0", dynamic_dim),
        "format": "NC1HWC0",
        "ori_shape": _trans_dynamic_shape(input_size, data_format, dynamic_dim),
        "ori_format": data_format,
        "dtype": dtype,
        "range": _get_range_from_shape(input_grad_shape, dynamic_dim)
    }
    strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]
    dilations = [1, dilation, dilation, 1] if data_format == "NHWC" else [1, 1, dilation, dilation]

    kernel_name = _get_kernel_name(filter_ori_shape, out_backprop_ori_shape, input_size, stride, dilation, pads, dtype)
    return {
        "params": [x, filter, out_backprop, input_grad, strides, dilations, pads, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


for case in dynamic_conv2d_bp_input_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_trans_data_case(case))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
