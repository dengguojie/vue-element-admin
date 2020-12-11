#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseConv2dBackpropFilter", "impl.depthwise_conv2d_backprop_filter_d",
               "depthwise_conv2d_backprop_filter_d")

dp_conv2d_bp_filter_op_testcase = [
    # fp16 without bias
    ((32, 960, 7, 7), (32, 960, 7, 7), (3, 3, 960, 1), 1, 1, (1, 1, 1, 1), "NCHW", "success"),
    ((3, 3, 960, 1), (32, 960, 7, 7), (32, 960, 7, 7), 1, 1, (1, 1, 1, 1, 1), "NCHW", RuntimeError),
    ((16, 192, 28, 28), (16, 192, 14, 14), (5, 5, 192, 1), 2, 1, (2, 2, 2, 2), "NCHW", "success"),
    ((8, 96, 56, 56), (8, 96, 56, 56), (3, 3, 96, 1), 1, 1, (1, 1, 1, 1), "NCHW", "success"),
    ((1, 32, 112, 112), (1, 32, 56, 56), (3, 3, 32, 1), 2, 1, (1, 1, 1, 1), "NCHW", "success"),
    ((2, 336, 504, 256), (2, 334, 502, 256), (3, 3, 256, 1), 1, 1, (0, 0, 0, 0), "NHWC", "success"),
    ((648, 4, 5, 2048), (648, 2, 3, 2048), (3, 3, 2048, 1), 1, 1, (0, 0, 0, 0), "NHWC", "success"),
    ((2, 28, 28, 240), (2, 14, 14, 240), (5, 5, 240, 1), 2, 1, (2, 2, 2, 2), "NHWC", "success"),
]


def _get_kernel_name(input_shape, out_backprop_shape, filter_grad_shape, stride, dilation, pads, dtype):
    kernel_name = 'dp_conv2d_bp_filter' + '_'.join(
        map(str, input_shape)) + '_' + '_'.join(map(str, out_backprop_shape)) + '_' + '_'.join(
            map(str, filter_grad_shape)) + '_' + str(stride) + '_' + str(dilation) + "_" + '_'.join(map(
                str, pads)) + '_' + dtype
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


def _gen_trans_data_case(param):
    input_shape, out_backprop_shape, filter_size, stride, dilation, pads, data_format, expect_result = param

    data_format = data_format.upper()
    dtype = "float16"
    filter_grad_dtype = "float32"

    input_fm = {
        "shape": _shape_to_NC1HWC0(input_shape, data_format, dtype),
        "format": "NC1HWC0",
        "ori_shape": input_shape,
        "ori_format": data_format,
        "dtype": dtype
    }
    out_backprop = {
        "shape": _shape_to_NC1HWC0(out_backprop_shape, data_format, dtype),
        "format": "NC1HWC0",
        "ori_shape": out_backprop_shape,
        "ori_format": data_format,
        "dtype": dtype
    }
    filter_grad = {
        "shape": _shape_to_C1HWNCoC0(filter_size, "HWCN", filter_grad_dtype),
        "ori_shape": filter_size,
        "ori_format": "HWCN",
        "format": "C1HWNCoC0",
        "dtype": filter_grad_dtype
    }
    strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]
    dilations = [1, dilation, dilation, 1] if data_format == "NHWC" else [1, 1, dilation, dilation]

    kernel_name = _get_kernel_name(input_fm, out_backprop, filter_grad, stride, dilation, pads, dtype)
    return {
        "params": [input_fm, out_backprop, filter_grad, filter_size, strides, dilations, pads, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


# ============ auto gen ["Ascend910"] test cases start ===============
for case in dp_conv2d_bp_filter_op_testcase:
    ut_case.add_case(["Ascend910"], _gen_trans_data_case(case))

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
