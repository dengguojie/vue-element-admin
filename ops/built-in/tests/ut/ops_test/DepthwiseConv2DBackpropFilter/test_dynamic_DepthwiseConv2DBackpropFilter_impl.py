#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseConv2DBackpropFilter", "impl.dynamic.depthwise_conv2d_backprop_filter",
               "depthwise_conv2d_backprop_filter")
dynamic_conv2d_bp_filter_op_testcase = [
    # success
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], "success", "dynamic_conv2d_bp_filter_op_testcase_0"),
    ((3, 200, 75, 40), (3, 200, 75, 40), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NHWC", [0, 2, 3], "success", "dynamic_conv2d_bp_filter_op_testcase_1"),
    ((3, 40, 200, 75), (3, 40, 192, 68), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0, 2, 3], "success", "dynamic_conv2d_bp_filter_op_testcase_2"),
    # dedy_c != fmap_c * filter_n
    ((3, 40, 200, 75), (3, 41, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_3"),
    # filter_c != fmap_c
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 13, 1), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_4"),
    # fmap_n != dedy_n
    ((3, 40, 200, 75), (4, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_5"),
    # stride_n != 1
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [2,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_6"),
    # strides dim != 4
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_7"),
    # pads dim != 4
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_8"),
    # stride_h/w > 63
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,64,64,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_9"),
    # dilations != [1,1,1,1]
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,2,2,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_10"),
    # dilations dim != 4
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_11"),
    # dedy_h does not match fmap_h
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_12"),
    # dedy_h less than 2
    ((3, 40, 9, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_13"),
    # -2
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0, 1, 2, 3], "success", "dynamic_conv2d_bp_filter_op_testcase_14"),
    # C -1
    ((3, -1, 200, 75), (3, -1, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0, 2, 3], "success", "dynamic_conv2d_bp_filter_op_testcase_15"),
    # dynamic_nh in input, dynamic_hw in dedy
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [[0,2],[2,3]], "success", "dynamic_conv2d_bp_filter_op_testcase_16"),
    # dynamic_w in input, dynamic_hw in dedy
    ((3, 40, 200, 75), (3, 40, 200, 75), (9, 9, 40, 1), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [[3],[2,3]], "success", "dynamic_conv2d_bp_filter_op_testcase_17"),
]


def _get_kernel_name(x_shape, dedy_shape, filter_shape, strides, dilations, pads, data_format):
    padding = "SAME" if -1 in pads else "VALID"
    kernel_name = 'dp_conv2d_bp_filter' + '_'.join(map(str, x_shape)) + '_' + '_'.join(map(str, dedy_shape)) + '_' + '_'.join(
        map(str, filter_shape)) + '_' + str(strides[0]) + '_' + str(dilations[0]) + "_" + padding + '_' + data_format
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


def _get_range_from_shape(shape, dynamic_dim):
    ori_range = [(dim, dim) for dim in shape]
    if dynamic_dim:
        for dim in dynamic_dim:
            ori_range[dim] = (max(1, shape[dim] // 2), min(4096, shape[dim] * 2))
    return ori_range


def _trans_dynamic_shape(shape, data_format, dynamic_dim):
    shape = list(shape)
    if 0 in dynamic_dim:
        n_dim = data_format.index("N")
        shape[n_dim] = -1
    if 2 in dynamic_dim:
        h_dim = data_format.index("H")
        shape[h_dim] = -1
    if 3 in dynamic_dim:
        w_dim = data_format.index("W")
        shape[w_dim] = -1
    return tuple(shape)


def _gen_case(param):
    input_ori_shape, out_backprop_ori_shape, filter_size, strides, dilations, pads, data_format, dynamic_dim, expect_result, kernel_name = param

    data_format = data_format.upper()
    dtype = "float16"

    unknown_rank_flag = False
    if len(dynamic_dim) == 4:
        unknown_rank_flag = True
        dynamic_dim = [0, 2, 3]

    input_shape = _shape_to_NC1HWC0(input_ori_shape, data_format, dtype)
    out_backprop_shape = _shape_to_NC1HWC0(out_backprop_ori_shape, data_format, dtype)
    filter_grad_shape = _shape_to_C1HWNCoC0(filter_size, "HWCN", dtype)
    x = {
        "shape": _trans_dynamic_shape(input_shape, "NC1HWC0", dynamic_dim[0] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "format": "NC1HWC0",
        "ori_shape": [-2] if unknown_rank_flag else _trans_dynamic_shape(input_ori_shape, data_format,
                     dynamic_dim[0] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "ori_format": data_format,
        "dtype": dtype,
        "range": _get_range_from_shape(input_shape, dynamic_dim[0] if isinstance(dynamic_dim[0], list) else dynamic_dim)
    }
    filter_shape = {
        "shape": [1,1,1,1],
        "ori_shape": [1,1,1,1],
        "ori_format": "HWCN",
        "format": "C1HWNCoC0",
        "dtype": "int32",
        "range": _get_range_from_shape(filter_size, [])
    }
    out_backprop = {
        "shape": _trans_dynamic_shape(out_backprop_shape, "NC1HWC0", dynamic_dim[1] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "format": "NC1HWC0",
        "ori_shape": _trans_dynamic_shape(out_backprop_ori_shape, data_format,
                     dynamic_dim[1] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "ori_format": data_format,
        "dtype": dtype,
        "range": _get_range_from_shape(out_backprop_shape, dynamic_dim[1] if isinstance(dynamic_dim[0], list) else dynamic_dim)
    }
    filter_grad = {
        "shape": filter_grad_shape,
        "format": "C1HWNCoC0",
        "ori_shape": filter_size,
        "ori_format": "HWCN",
        "dtype": "float32",
        "range": _get_range_from_shape(filter_grad_shape, [])
    }

    kernel_name = kernel_name
    return {
        "params": [x, filter_shape, out_backprop, filter_grad, strides, dilations, pads, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


for case in dynamic_conv2d_bp_filter_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_case(case))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
