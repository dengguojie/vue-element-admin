#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2DBackpropInput", "impl.dynamic.conv2d_backprop_input",
               "conv2d_backprop_input")

dynamic_conv2d_bp_input_op_testcase = [
    ((3, 3, 32, 32), (32, 32, 7, 7), (32, 32, 7, 7), (1, 1), (1, 1, 1, 1), "NCHW", 1, [0], "success"),
    ((5, 5, 192, 192), (1, 192, 14, 28), (1, 192, 28, 28), (2, 2), (-1, -1, -1, -1), "NCHW", 1, [0, 2], "success"),
    ((3, 3, 96, 96), (8, 96, 56, 56), (8, 96, 56, 56), (1, 1), (-1, -1, -1, -1), "NCHW", 1, [0, 3], "success"),
    ((1, 1, 32, 32), (1, 32, 28, 55), (1, 32, 55, 55), (2, 2), (0, 0, 0, 0), "NCHW", 1, [2, 3], "success"),
    ((3, 3, 256, 256), (2, 34, 32, 256), (2, 36, 34, 256), (1, 1), (0, 0, 0, 0), "NHWC", 1, [0, 2, 3], "success"),
    ((5, 5, 240, 240), (2, 14, 14, 240), (2, 28, 28, 240), (2, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 2, 3], "success"),
    ((3, 3, 16, 16), (2, 5, 5, 32), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 2, 3], RuntimeError),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [1], RuntimeError),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 1], "success"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [1, 2], "success"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [1, 3], "success"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 1, 2, 3], "success"),
    ((3, 3, 16, 16), [-2], (1, 16, 5, 5), (1, 1), (-1, -1, -1, -1), "NCHW", 1, [0, 1, 2, 3], "success"),
    ((7, 6, 64, 10), [-2], (1, 64, 5, 5), (1, 1), (-1, -1, -1, -1), "NCHW", 1, [0, 1, 2, 3], "success"),
    ((1, 1, 32, 32), (1, 32, 4096, 4096), (1, 32, 4096, 4096), (1, 1), (100, 100, 100, 100), "NCHW", 1, [2, 3], "success"),
]


def _get_kernel_name(filter_shape, dy_shape, x_shape, strides, pads, dilation, dtype):
    padding = "SAME" if -1 in pads else "VALID"
    if dy_shape == [-2]:
        dy_shape_info = "neg_2"
    else:
        dy_shape_info = '_'.join(map(str, dy_shape))
    kernel_name = 'dynamic_conv2d_backprop_input_' + '_'.join(map(str, filter_shape)) + '_' + dy_shape_info + '_' + '_'.join(
        map(str, x_shape)) + '_' + '_'.join(map(str, strides)) + '_' + '_'.join(
        map(str, dilation)) + "_" + padding + '_' + dtype
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


def _trans_dynamic_shape(shape, format, dynamic_dim, tran_flag=False):
    shape = list(shape)
    if len(format) == 4:
        if 0 in dynamic_dim:
            n_dim = format.index("N")
            shape[n_dim] = -1
        if 1 in dynamic_dim and tran_flag:
            c_dim = format.index("C")
            shape[c_dim] = -1
        if 2 in dynamic_dim:
            h_dim = format.index("H")
            shape[h_dim] = -1
        if 3 in dynamic_dim:
            w_dim = format.index("W")
            shape[w_dim] = -1
    else:
        if 0 in dynamic_dim:
            shape[0] = -1
        if 1 in dynamic_dim and tran_flag:
            shape[1] = -1
        if 2 in dynamic_dim:
            shape[2] = -1
        if 3 in dynamic_dim:
            shape[3] = -1
    return tuple(shape)


def _gen_trans_data_case(param):
    filter_ori_shape, out_backprop_ori_shape, input_size, strides, pads, data_format, group, dynamic_dim, expect_result = param
    dilations = (1, 1, 1, 1)
    dtype = "float16"
    data_format = data_format.upper()
    if out_backprop_ori_shape == [-2]:
        input_size_op = [-1, input_size[data_format.index("C")], -1, -1]
        x = {'shape': [4], 'format': 'NC1HWC0', 'ori_shape': [4], 'ori_format': 'NHWC', 'dtype': 'float16',
             'range': [(1, 4), (1, 2), (2, 10), (2, 10), (16, 16)]}
        filter = {'shape': _shape_to_C1HWNCoC0(filter_ori_shape, "HWCN", dtype), 'ori_shape': filter_ori_shape, 'ori_format': 'HWCN',
                  'format': 'C1HWNCoC0', 'dtype': 'float16', 'range': None}
        out_backprop = {'shape': out_backprop_ori_shape, 'format': 'NC1HWC0', 'ori_shape': out_backprop_ori_shape,
                        'ori_format': "NCHW", 'dtype': 'float16', 'range': None}
        dx = {'shape': _shape_to_NC1HWC0(input_size_op, data_format, dtype), 'format': 'NC1HWC0',
                      'ori_shape': input_size_op, 'ori_format': "NCHW", 'dtype': 'float16', 'range': None}
    else:
        filter_shape = _shape_to_C1HWNCoC0(filter_ori_shape, "HWCN", dtype),
        out_backprop_shape = _shape_to_NC1HWC0(out_backprop_ori_shape, data_format, dtype)
        dx_shape = _shape_to_NC1HWC0(input_size, data_format, dtype)
        x = {
            "shape": [4],
            "format": "NC1HWC0",
            "ori_shape": [4],
            "ori_format": data_format,
            "dtype": dtype,
            "range": _get_range_from_shape(dx_shape, dynamic_dim)
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
            "shape": _trans_dynamic_shape(out_backprop_shape, "NC1HWC0", dynamic_dim, True),
            "format": "NC1HWC0",
            "ori_shape": _trans_dynamic_shape(out_backprop_ori_shape, data_format, dynamic_dim, True),
            "ori_format": data_format,
            "dtype": dtype,
            "range": _get_range_from_shape(out_backprop_shape, dynamic_dim)
        }
        dx = {
            "shape": _trans_dynamic_shape(dx_shape, "NC1HWC0", dynamic_dim),
            "format": "NC1HWC0",
            "ori_shape": _trans_dynamic_shape(input_size, data_format, dynamic_dim),
            "ori_format": data_format,
            "dtype": dtype,
            "range": _get_range_from_shape(dx_shape, dynamic_dim)
        }
    stride_h, stride_w = strides
    strides = [1, stride_h, stride_w, 1] if data_format == "NHWC" else [1, 1, stride_h, stride_w]

    kernel_name = _get_kernel_name(filter_ori_shape, out_backprop_ori_shape, input_size, strides, pads, dilations, dtype)

    print(x, filter, out_backprop, dx)
    return {
        "params": [x, filter, out_backprop, dx, strides, pads, dilations, group, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


for case in dynamic_conv2d_bp_input_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_trans_data_case(case))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
