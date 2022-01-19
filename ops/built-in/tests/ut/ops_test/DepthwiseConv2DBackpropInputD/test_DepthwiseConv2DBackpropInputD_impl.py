#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseConv2dBackpropInputD", "impl.depthwise_conv2d_backprop_input_d",
               "depthwise_conv2d_backprop_input_d")

dp_conv2d_bp_input_op_testcase = [
    # fp16 without bias
    ((960, 1, 3, 3), (32, 960, 7, 7), (32, 960, 7, 7), 1, 1, (1, 1, 1, 1), "NCHW", "success"),
    ((192, 1, 5, 5), (16, 192, 1, 1), (16, 192, 2, 2), 2, 1, (2, 2, 2, 2), "NCHW", "success"),
    ((96, 1, 3, 3), (8, 96, 1, 1), (8, 96, 3, 3), 1, 1, (0, 0, 0, 0), "NCHW", "success"),
    ((32, 1, 3, 3), (1, 32, 2, 2), (1, 32, 5, 5), 2, 1, (0, 0, 0, 0), "NCHW", "success"),
    ((3, 3, 256, 1), (2, 334, 502, 1), (2, 336, 504, 256), 2, 1, (0, 0, 0, 0, 0), "NHWC", RuntimeError),
    ((3, 3, 256, 1), (2, 334, 502, 256), (2, 336, 504, 256), 2, 2, (0, 0, 0, 0), "NHWC", RuntimeError),
    ((2, 32, 3, 3), (1, 32, 2, 2), (1, 32, 5, 5), 2, 1, (0, 0, 0, 0), "NCHW", RuntimeError),
    ((1, 16, 3, 3), (1, 32, 2, 2), (1, 32, 5, 5), 2, 1, (0, 0, 0, 0), "NCHW", RuntimeError),
    ((2, 32, 3, 3), (1, 32, 2, 2), (2, 32, 5, 5), 2, 1, (0, 0, 0, 0), "NCHW", RuntimeError),
    ((2, 32, 3, 3), (1, 16, 2, 2), (1, 32, 5, 5), 2, 1, (0, 0, 0, 0), "NCHW", RuntimeError),
    ((3, 3, 1, 2048), (648, 2, 3, 2048), (648, 4, 5, 2048), 1, 1, (0, 0, 0, 0), "NHWC", "success"),
    ((5, 5, 1, 240), (2, 14, 14, 240), (2, 28, 28, 240), 2, 1, (2, 2, 2, 2), "NHWC", "success"),
]

def _lcm(param1, param2):
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, param1 % param2

    return temp // param2


def ceil(x_1, x_2):
    return (x_1 + x_2 - 1) // x_2


def calculate_group(cin, cout, groups):
    
    dx_c_ori = cin
    dy_c_ori = cout // groups

    c0_size = 16
    c0_size_k = 16

    dx_c_extend = _lcm(dx_c_ori, c0_size) // dx_c_ori
    dy_c_extend = _lcm(dy_c_ori, c0_size_k) // dy_c_ori
    multiple_extend = min(_lcm(dx_c_extend, dy_c_extend), groups)

    dx_c1_extend = ceil(multiple_extend * dx_c_ori, c0_size)
    dy_c1_extend = ceil(multiple_extend * dy_c_ori, c0_size_k)

    group_extend = ceil(groups, multiple_extend)
    
    return group_extend, dx_c1_extend, dy_c1_extend

def _get_kernel_name(x_shape, filter_shape, bias_shape, stride, dilation, pads):
    bias_shape = bias_shape if bias_shape else []
    kernel_name = 'dp_conv2dbp_input_' + '_'.join(map(str, x_shape)) + '_' + '_'.join(map(str, filter_shape)) + '_' + '_'.join(
        map(str, bias_shape)) + '_' + str(stride) + '_' + str(dilation) + "_" + '_'.join(map(str, pads))
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
    real_g, cin1_g, cout1_g = calculate_group(c, n, n)
    return (real_g * cin1_g * h * w, cout1_g, 16, c0)


def _gen_trans_data_case(param):
    filter_shape, out_backprop_shape, input_size, stride, dilation, pads, data_format, expect_result = param

    data_format = data_format.upper()
    dtype = "float16"
    filter_format = 'NCHW' if data_format == 'NCHW' else 'HWCN'

    filter = {
        "shape": _shape_to_C1HWNCoC0(filter_shape, filter_format, dtype),
        "ori_shape": filter_shape,
        "ori_format": filter_format,
        "format": "C1HWNCoC0",
        "dtype": dtype
    }
    out_backprop = {
        "shape": _shape_to_NC1HWC0(out_backprop_shape, data_format, dtype),
        "format": "NC1HWC0",
        "ori_shape": out_backprop_shape,
        "ori_format": data_format,
        "dtype": dtype
    }
    input_grad = {
        "shape": _shape_to_NC1HWC0(input_size, data_format, dtype),
        "format": "NC1HWC0",
        "ori_shape": input_size,
        "ori_format": data_format,
        "dtype": dtype
    }
    strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]
    dilations = [1, dilation, dilation, 1] if data_format == "NHWC" else [1, 1, dilation, dilation]

    kernel_name = _get_kernel_name(filter_shape, out_backprop_shape, input_size, stride, dilation, pads)
    return {
        "params": [filter, out_backprop, input_grad, input_size, strides, dilations, pads, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


# ============ auto gen ["Ascend910"] test cases start ===============
for case in dp_conv2d_bp_input_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_trans_data_case(case))

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
