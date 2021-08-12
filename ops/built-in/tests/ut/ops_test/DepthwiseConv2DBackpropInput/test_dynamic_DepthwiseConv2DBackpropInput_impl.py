#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseConv2DBackpropInput", "impl.dynamic.depthwise_conv2d_backprop_input",
               "depthwise_conv2d_backprop_input")
dynamic_depthwise_conv2d_bp_input_op_testcase = [
    ((1, 192, 5, 5), (16, 192, -1, 28), (16, 192, -1, 28), (2, 2), (-1, -1, -1, -1), "NCHW", [0, 2], None, "success", "depthwise_conv2d_bp_input_w_dim_upper_boud_None"),
    ((3, 3, 256, 1), (2, 334, 502, 256), (2, 336, 504, 256), (1, 1), (0, 0, 0, 0), "NHWC", [0, 2, 3], None, "success", "depthwise_conv2d_bp_input_dynamic_nhw_pad_valid"),
    ((5, 5, 240, 1), (2, 1, 1, 240), (2, 2, 2, 240), (2, 2), (-1, -1, -1, -1), "NHWC", [0, 2, 3], None, "success", "depthwise_conv2d_bp_input_dynamic_nhw_pad_same"),
    ((3, 3, 16, 1), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", [0, 1], None, "success", "depthwise_conv2d_bp_input_dynamic_nc"),
    ((3, 3, 17, 1), (2, 5, 5, 17), (2, 5, 5, 17), (1, 1), (-1, -1, -1, -1), "NHWC", [1, 3], None, "success", "depthwise_conv2d_bp_input_dynamic_cw"),
    ((2, 2, 16, 1), (2, 2, 1, 16), (2, 4, 2, 16), (2, 1), (0, 0, 0, 0), "NHWC", [2], None, "success", "depthwise_conv2d_bp_input_dynamic_h"),
    ((1, 96, 3, 3), (-1, 96, 2, -1), (-1, 96, 2, -1), (1, 2), (-1, -1, -1, -1), "NCHW", [0, 3], None, "success", "depthwise_conv2d_bp_input_nw_dim_upper_boud_None"),
    ((3, 3, 16, 1), (2, 1, 2, 16), (2, 3, 5, 16), (1, 2), (0, 0, 0, 0), "NHWC", [0, 1, 2, 3], None, "success", "depthwise_conv2d_bp_input_dynamic_nchw_pad_valid"),
    ((1, 960, 3, 3), (32, 960, 7, 7), (32, 960, 7, 7), (1, 1), (1, 1, 1, 1), "NCHW", [0], None, "success", "depthwise_conv2d_bp_input_dynamic_n"),
    ((1, 1, 32, 1), (1, 1, 1, 32), (1, 2, 2, 32), (2, 2), (0, 0, 0, 0), "NHWC", [2, 3], None, "success", "depthwise_conv2d_bp_input_dynamic_hw"),
    ((1, 16, 3, 3), [-2], (1, 16, 5, 5), (1, 1), (-1, -1, -1, -1), "NCHW", [0, 1, 2, 3], None, "success", "depthwise_conv2d_bp_input_unknown_rank"),
    
    ((3, 3, 16, 2), (2, 3, 3, 16), (2, 5, 5, 16), (2, 2), (-1, -1, -1, -1), "NHWC", [0, 2, 3], None, RuntimeError, "depthwise_conv2d_bp_input_dedx_c_not_equal_filer"),
    ((3, 3, 16, 1), (2, 5, 5, 32), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", [1], None, RuntimeError, "depthwise_conv2d_bp_input_dedy_nhw_large_than_1"),
    ((1, 1, 14, 20), (1, 1, 28, 3507), (1, 1, 111, 3507), (4, 1), (-1, -1, -1, -1), "NCHW", [0, 2, 3], [[(1, 1),(1, 1), (27, 27), (2143, 2143)], [(1, 1),(1, 1), (120, 120), (2157, 2157)]], RuntimeError, "depthwise_conv2d_bp_input_dynamic_nhw_large_than_l1size"),
]


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


def _get_range_from_shape(shape, dynamic_dim=None):
    ori_range = [(dim, dim) for dim in shape]
    if dynamic_dim:
        for dim in dynamic_dim:
            if shape[dim] == -1:
                ori_range[dim] = (max(1, shape[dim] // 2), None)
            else:
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
    filter_ori_shape, out_backprop_ori_shape, input_size, strides, pads, data_format, dynamic_dim, dy_dx_range, expect_result, case_name = param

    dilations = (1, 1, 1, 1)
    dtype = "float16"
    data_format = data_format.upper()
    filter_format = 'NCHW' if data_format == 'NCHW' else 'HWCN'
    if out_backprop_ori_shape == [-2]:
        input_size_op = [-1, input_size[data_format.index("C")], -1, -1]
        x = {'shape': [4], 'format': 'NC1HWC0', 'ori_shape': [4], 'ori_format': 'NHWC', 'dtype': 'float16',
             'range': [(1, 4), (1, 2), (2, 10), (2, 10), (16, 16)]}
        filter = {'shape': ((1, 3, 3, 1, 16, 16),), 'ori_shape': (3, 3, 16, 1), 'ori_format': 'HWCN',
                  'format': 'C1HWNCoC0', 'dtype': 'float16', 'range': None}
        out_backprop = {'shape': out_backprop_ori_shape, 'format': 'NC1HWC0', 'ori_shape': out_backprop_ori_shape,
                        'ori_format': "NCHW", 'dtype': 'float16', 'range': None}
        input_grad = {'shape': _shape_to_NC1HWC0(input_size_op, data_format, dtype), 'format': 'NC1HWC0',
                      'ori_shape': input_size_op, 'ori_format': "NCHW", 'dtype': 'float16', 'range': None}
    else:
        filter_shape = _shape_to_C1HWNCoC0(filter_ori_shape, filter_format, dtype),
        out_backprop_shape = _shape_to_NC1HWC0(out_backprop_ori_shape, data_format, dtype)
        input_grad_shape = _shape_to_NC1HWC0(input_size, data_format, dtype)
        x_range = _get_range_from_shape(input_grad_shape, dynamic_dim)
        filter_range = _get_range_from_shape(filter_shape)
        out_backprop_range = _get_range_from_shape(out_backprop_shape, dynamic_dim)
        y_range = _get_range_from_shape(input_grad_shape, dynamic_dim)
        if dy_dx_range:
            out_backprop_range = dy_dx_range[0]
            y_range = dy_dx_range[1]
        x = {
            "shape": [4],
            "format": "NC1HWC0",
            "ori_shape": [4],
            "ori_format": data_format,
            "dtype": dtype,
            "range": x_range
        }
        filter = {
            "shape": filter_shape,
            "ori_shape": filter_ori_shape,
            "ori_format": filter_format,
            "format": "C1HWNCoC0",
            "dtype": dtype,
            "range": filter_range
        }
        out_backprop = {
            "shape": _trans_dynamic_shape(out_backprop_shape, "NC1HWC0", dynamic_dim, True),
            "format": "NC1HWC0",
            "ori_shape": _trans_dynamic_shape(out_backprop_ori_shape, data_format, dynamic_dim, True),
            "ori_format": data_format,
            "dtype": dtype,
            "range": out_backprop_range
        }
        input_grad = {
            "shape": _trans_dynamic_shape(input_grad_shape, "NC1HWC0", dynamic_dim),
            "format": "NC1HWC0",
            "ori_shape": _trans_dynamic_shape(input_size, data_format, dynamic_dim),
            "ori_format": data_format,
            "dtype": dtype,
            "range": y_range
        }
    stride_h, stride_w = strides
    strides = [1, stride_h, stride_w, 1] if data_format == "NHWC" else [1, 1, stride_h, stride_w]

    return {
        "params": [x, filter, out_backprop, input_grad, strides, dilations, pads, data_format],
        "case_name": case_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


for case in dynamic_depthwise_conv2d_bp_input_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_trans_data_case(case))


def test_depthwise_conv2d_backprop_input_fuzz_build_generalization_general(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (11, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(16, 32), (3, 3), (8, 16), (8, 16), (16, 16)],
            'ori_range': [(16, 32), (33, 33), (8, 16), (8, 16)]
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'depthwise_conv2d_backprop_input_fuzz_build_generalization_general']
    depthwise_conv2d_backprop_input_generalization(*input_list)


ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_generalization_general)


def test_depthwise_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 26, 2888, 16),
            'ori_shape': (50, 2, 26, 2888),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(32, 64), (1, 1), (16, 32), (1024, 4096), (16, 16)],
            'ori_range': [(32, 64), (2, 2), (16, 32), (1024, 4096)]
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'depthwise_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed']
    depthwise_conv2d_backprop_input_generalization(*input_list)


ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed)


def test_depthwise_conv2d_backprop_input_fuzz_build_generalization_h_range_max_fixed(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (1, 1, 16, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (3, 1, 1, 3276, 16),
            'ori_shape': (3, 1, 1, 3276),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(1, 3), (1, 1), (1, 3), (1024, 4096), (16, 16)],
            'ori_range': [(1, 3), (1, 1), (1, 3), (1024, 4096)]
        }, {
            'shape': (3, 1, 3, 3276, 16),
            'ori_shape': (3, 1, 3, 3276),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 4, 1), (1, 1, 1, 1), (6, 7, 4, 5), 'NCHW',
        'depthwise_conv2d_backprop_input_fuzz_build_generalization_h_range_max_fixed']
    depthwise_conv2d_backprop_input_generalization(*input_list)


ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_generalization_h_range_max_fixed)


def test_depthwise_conv2d_backprop_input_fuzz_build_support_mode_error(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (11, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(16, 32), (3, 3), (8, 16), (8, 16), (16, 16)],
            'ori_range': [(16, 32), (33, 33), (8, 16), (8, 16)]
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_input_fuzz_build_support_mode_error', {"mode": "mode"}]
    try:
        depthwise_conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("support mode is error")


def test_depthwise_conv2d_backprop_input_fuzz_build_no_support_neg_two(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (11, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-2,),
            'ori_shape': (-2,),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(16, 32), (3, 3), (8, 16), (8, 16), (16, 16)],
            'ori_range': [(16, 32), (33, 33), (8, 16), (8, 16)]
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_input_fuzz_build_no_support_neg_two']
    try:
        depthwise_conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("not support unknwon_rank")


def test_depthwise_conv2d_backprop_input_fuzz_build_ori_format_error(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (11, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'ND',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(16, 32), (3, 3), (8, 16), (8, 16), (16, 16)],
            'ori_range': [(16, 32), (33, 33), (8, 16), (8, 16)]
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_input_fuzz_build_ori_format_error']
    try:
        depthwise_conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("not support ori_format")


def test_depthwise_conv2d_backprop_input_fuzz_build_ori_shape_error(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (11, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14),
            'ori_format': 'ND',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(16, 32), (3, 3), (8, 16), (8, 16), (16, 16)],
            'ori_range': [(16, 32), (33, 33), (8, 16), (8, 16)]
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_input_fuzz_build_ori_shape_error']
    try:
        depthwise_conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("not support ori_shape")


def test_depthwise_conv2d_backprop_input_fuzz_build_shape_error(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (11, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (16, 3, 14, 12),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(16, 32), (3, 3), (8, 16), (8, 16), (16, 16)],
            'ori_range': [(16, 32), (33, 33), (8, 16), (8, 16)]
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_input_fuzz_build_shape_error']
    try:
        depthwise_conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("not support shape")

def test_depthwise_conv2d_backprop_input_fuzz_build_tilingcase(test_arg):
    import json
    from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input
    from tbe.common.context import get_context
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        get_context().set_build_type("fuzzily_build")
        get_context().add_addition("max_kernel_id", -1)
        missing_info = [{
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [1, 1],
                                        [5, 5],
                                        [4, 15],
                                        [641, 767]
                                    ],
                                    "shape": [-1, 5, -1, -1]
                                }]
                            }]
                        }, {
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [1, 1],
                                        [5, 5],
                                        [5, 15],
                                        [512, 640]
                                    ],
                                    "shape": [-1, 5, -1, -1]
                                }]
                            }]
                        }]
        get_context().add_addition("missing_support_info", json.dumps(missing_info))
        input_list = [
            {
                'shape': (4,),
                'ori_shape': (4,),
                'ori_format': 'ND',
                'format': 'ND',
                'dtype': 'int32',
                'range': ()
            }, {
                'shape': (1, 5, 6, 14),
                'ori_shape': (1, 5, 6, 14),
                'ori_format': 'NCHW',
                'format': 'NCHW',
                'dtype': 'float16'
            }, {
                'shape': (-1, 1, -1, -1, 16),
                'ori_shape': (-1, 5, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16',
                'range': ((1, 1), (1, 1), (4, 15), (512, 767), (16, 16))
            }, {
                'shape': (-1, 1, -1, -1, 16),
                'ori_shape': (-1, 5, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16',
                'range': ((1, 1), (1, 1), (18, 65), (1036, 1547), (16, 16))
            }, (1, 1, 4, 2), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW', 'test_conv2d_fuzz_build_tilingcase']
        depthwise_conv2d_backprop_input(*input_list)

ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_tilingcase)
ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_support_mode_error)
ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_no_support_neg_two)
ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_ori_format_error)
ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_ori_shape_error)
ut_case.add_cust_test_func(test_func=test_depthwise_conv2d_backprop_input_fuzz_build_shape_error)


if __name__ == '__main__':
    ut_case.run()
    exit(0)
