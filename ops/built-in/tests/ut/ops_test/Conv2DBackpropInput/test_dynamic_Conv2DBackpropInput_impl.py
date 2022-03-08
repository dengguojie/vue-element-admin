#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from numpy import var
from te import tvm
from tbe.dsl.base import operation
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.common.context.op_context import OpContext
from impl.dynamic.conv2d_backprop_input import get_op_support_info
from impl.dynamic.conv2d_backprop_input import conv2dbp_input_fusion_compute
from impl.dynamic.trans_data import trans_data_fusion_compute
from tbe.common.utils import shape_util

ut_case = OpUT("Conv2DBackpropInput", "impl.dynamic.conv2d_backprop_input",
               "conv2d_backprop_input")

dynamic_conv2d_bp_input_op_testcase = [
    ((192, 192, 5, 5), (1, 192, -1, 28), (1, 192, -1, 28), (2, 2), (-1, -1, -1, -1), "NCHW", 1, [0, 2], "success", "conv2d_bp_input_w_dim_upper_boud_None"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 1], "success", "conv2d_bp_input_dynamic_nc"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [1, 3], "success", "conv2d_bp_input_dynamic_cw"),
    ((3, 3, 16, 16), (2, 1, 1, 16), (2, 4, 3, 16), (2, 1), (0, 0, 0, 0), "NHWC", 1, [1, 2], "success", "conv2d_bp_input_dynamic_ch"),
    ((96, 96, 3, 3), (-1, 96, 2, -1), (-1, 96, 2, -1), (1, 2), (-1, -1, -1, -1), "NCHW", 1, [0, 3], "success", "conv2d_bp_input_nw_dim_upper_boud_None"),
    ((3, 3, 256, 256), (2, 34, 32, 256), (2, 36, 34, 256), (1, 1), (0, 0, 0, 0), "NHWC", 1, [0, 2, 3], "success", "conv2d_bp_input_dynamic_nhw_padding_valid"),
    ((5, 5, 240, 240), (2, 1, 1, 240), (2, 2, 1, 240), (2, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 2, 3], "success", "conv2d_bp_input_dynamic_nhw_padding_same"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 1, 2, 3], "success", "conv2d_bp_input_dynamic_nchw_padding_same"),
    ((32, 32, 3, 3), (32, 32, 7, 1), (32, 32, 7, 2), (1, 2), (1, 1, 1, 1), "NCHW", 1, [0], "success", "conv2d_bp_input_dynamic_n"),
    ((1, 32, 1, 1), (1, 1, 4096, 4096), (1, 32, 4096, 4096), (1, 1), (100, 100, 100, 100), "NCHW", 1, [2, 3], "success", "conv2d_bp_input_dynamic_hw"),
    ((1, 1, 55, 2), (1, 16, 55, 2), (1, 32, 55, 55), (2, 1), (0, 0, 0, 0), "NHWC", 1, [2, 3], "success", "conv2d_bp_input_dynamic_hw_padding_valid"),
    ((16, 7, 3, 3), [-2], (1, 7, 3, 3), (2, 2), (-1, -1, -1, -1), "NCHW", 1, [0, 1, 2, 3], "success", "conv2d_bp_input_unknown_rank_padding_same"),
    ((7, 6, 64, 10), [-2], (1, 7, 6, 64), (1, 1), (0, 0, 0, 0), "NHWC", 1, [0, 1, 2, 3], "success", "conv2d_bp_input_unknown_rank_padding_valid"),
    ((128, 256, 1, 1), (32, 128, 56, 56), (32, 256, 56, 56), (1, 1), (0, 0, 0, 0), "NCHW", 1, [2, 3], "success", "conv2d_bp_input_opti_dynamic_hw"),

    ((3, 3, 16, 16), (2, 5, 5, 32), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 2, 3], RuntimeError, "conv2d_bp_input_dedy_c_not_equal_filer"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [1], RuntimeError, "conv2d_bp_input_dedy_nhw_large_than_1"),
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
    filter_ori_shape, out_backprop_ori_shape, input_size, strides, pads, data_format, group, dynamic_dim, expect_result, case_name = param
    dilations = (1, 1, 1, 1)
    dtype = "float16"
    data_format = data_format.upper()
    filter_format = 'NCHW' if data_format == 'NCHW' else 'HWCN'
    if out_backprop_ori_shape == [-2]:
        input_size_op = [-1, input_size[data_format.index("C")], -1, -1] if data_format == 'NCHW' else [-1, -1, -1, input_size[data_format.index("C")]]
        x = {'shape': [4], 'format': 'NC1HWC0', 'ori_shape': [4], 'ori_format': data_format, 'dtype': 'float16',
             'range': [(1, 4), (1, 2), (2, 10), (2, 10), (16, 16)]}
        filter = {'shape': _shape_to_C1HWNCoC0(filter_ori_shape, filter_format, dtype), 'ori_shape': filter_ori_shape, 'ori_format': filter_format,
                  'format': 'C1HWNCoC0', 'dtype': 'float16', 'range': None}
        out_backprop = {'shape': out_backprop_ori_shape, 'format': 'NC1HWC0', 'ori_shape': out_backprop_ori_shape,
                        'ori_format': data_format, 'dtype': 'float16', 'range': None}
        dx = {'shape': _shape_to_NC1HWC0(input_size_op, data_format, dtype), 'format': 'NC1HWC0',
                      'ori_shape': input_size_op, 'ori_format': data_format, 'dtype': 'float16', 'range': None}
    else:
        filter_shape = _shape_to_C1HWNCoC0(filter_ori_shape, filter_format, dtype),
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
            "ori_format": filter_format,
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

    return {
        "params": [x, filter, out_backprop, dx, strides, pads, dilations, group, data_format],
        "case_name": case_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


for case in dynamic_conv2d_bp_input_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_trans_data_case(case))

test_case_error_format = _gen_trans_data_case(((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 1], RuntimeError, "test_case_error_format"))
test_case_error_format["params"][3]["ori_format"] = "NCHW"
ut_case.add_case(["Ascend910A"], test_case_error_format)

def test_conv2d_backprop_input_fuzz_build_generalization_general(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (16, 3, 16, 16)
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_general', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_generalization_general)


def test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
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
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed)

def test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_0(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 3200)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 26, 3191, 16),
            'ori_shape': (50, 2, 26, 3191),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 35, 3200, 16),
            'ori_shape': (50, 2, 35, 3200),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_0', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_0)

def conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_1(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
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
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_1', {"mode": "keep"}]
    try:
        conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("invalid generalize mode")


ut_case.add_cust_test_func(test_func=conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_1)

def conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_2(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 26, 2888, 16),
            'ori_shape': (-2,),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_2', {"mode": "keep_rank"}]
    try:
        conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("not support unknow_rank under mode keep_rank")

ut_case.add_cust_test_func(test_func=conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_2)

def conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_3(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 26, 2888, 16),
            'ori_shape': (50, 2, 26, 2888),
            'ori_format': 'HWCN',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_3', {"mode": "keep_rank"}]
    try:
        conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("invalid inputs ori_format")

ut_case.add_cust_test_func(test_func=conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_3)

def conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_4(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 26, 2888, 16),
            'ori_shape': (50, 2, 26),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_4', {"mode": "keep_rank"}]
    try:
        conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("invalid inputs ori_shape")

ut_case.add_cust_test_func(test_func=conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_4)

def test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_5(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 26, 16),
            'ori_shape': (50, 2, 26, 2888),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_5', {"mode": "keep_rank"}]
    try:
        conv2d_backprop_input_generalization(*input_list)
    except RuntimeError:
        print("invalid inputs shape")

ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_5)

def test_conv2d_backprop_input_fuzz_build_lower_limit(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (16, 3, 16, 16)
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-1, 3, -1, -1, 16),
            'ori_shape': (-1, 33, -1, -1),
            'range': ((16, 31), (3, 3), (4, 15), (4, 15), (16, 16)),
            'ori_range': ((16, 31), (33, 33), (4, 15), (4, 15)),
            'ori_format': 'NHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 3, -1, -1),
            'range': ((16, 31), (1, 1), (16, 31), (16, 31), (16, 16)),
            'ori_range': ((16, 31), (3, 3), (16, 31), (16, 31)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_general', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_lower_limit)

def test_conv2d_backprop_input_fuzz_build_upper_limit(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 2, -1, -1),
            'range': ((32, 2**31 - 1), (1, 1), (16, 31), (1024, 4096), (16, 16)),
            'ori_range': ((32, 2**31 - 1), (2, 2), (16, 31), (1024, 4096)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 2, -1, -1),
            'range': ((32, 2**31 - 1), (1, 1), (32, 63), (1024, 4096), (16, 16)),
            'ori_range': ((32, 2**31 - 1), (2, 2), (32, 63), (1024, 4096)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed_case', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)
ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_upper_limit)

def test_conv2d_backprop_input_fuzz_build_gragh_mode_pass(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (16, 3, 16, 16)
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-1, 3, -1, -1, 16),
            'ori_shape': (-1, 33, -1, -1),
            'range': ((16, 31), (3, 3), (4, 15), (4, 15), (16, 16)),
            'ori_range': ((16, 31), (33, 33), (4, 15), (4, 15)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 3, -1, -1),
            'range': ((16, 31), (1, 1), (16, 31), (16, 31), (16, 16)),
            'ori_range': ((16, 31), (3, 3), (16, 31), (16, 31)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_general', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_gragh_mode_pass)

def test_get_op_support_info_dynamic_dx_0(test_arg):
    y = {"shape": (-1, 4, -1, -1, 16), 'ori_shape': (-1, -1, -1, 64),
         "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
         "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
        }
    out_backprop = {"shape":  (-1, 4, -1, -1, 16), 'ori_shape':(-1, -1, -1, 64),
                    "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
                    "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
                   }
    filter = {"shape":  (36, 4, 16, 16), 'ori_shape':(3, 3, 64, 64),
              "ori_format": "NHWC", "format": "FRATAL_NZ", "dtype": "float16",
              "range": ((36, 36), (4, 4), (16, 16), (16, 16))
             }
    input_size =  (-1, -1, -1, 64)
    get_op_support_info(input_size, filter, out_backprop, y, (1, 1, 1, 1), (0, 0, 0, 0))
def test_get_op_support_info_dynamic_dx_1(test_arg):
    y = {"shape": (-1, 4, -1, -1, 16), 'ori_shape': (-1, -1, -1, 64),
         "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
         "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
        }
    out_backprop = {"shape":  (-1, 4, -1, -1, 16), 'ori_shape':(-1, -1, -1, 64),
                    "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
                    "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
                   }
    filter = {"shape":  (4, 4, 16, 16), 'ori_shape':(1, 1, 64, 64),
              "ori_format": "NHWC", "format": "FRATAL_NZ", "dtype": "float16",
              "range": ((4, 4), (4, 4), (16, 16), (16, 16))
             }
    input_size =  (-1, -1, -1, 64)
    get_op_support_info(input_size, filter, out_backprop, y, (1, 1, 1, 1), (0, 0, 0, 0))
ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_dx_0)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_dx_1)


def _build_dx_transdata_fusion_op(tensor_list, y, stride, dy_tensor, case_name):
    pads = [-1, -1, -1, -1]
    conv_res = conv2dbp_input_fusion_compute(
        tensor_list[0], tensor_list[1], dy_tensor, y, stride, pads, data_format="NCHW", kernel_name=case_name)
    tran_data_res = trans_data_fusion_compute(conv_res, {"ori_shape": y.get("ori_shape")}, "NC1HWC0", "NCHW")

    tensor_list.append(tran_data_res)
    with tvm.target.cce():
        sch = auto_schedule(tran_data_res)
    config = {
        "name": case_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extend_in_infer_bound": False}
    }
    build(sch, config)


def _test_dx_transdata_fusion_op(test_args):
    """
    dx + transdata
    dy: nc1hwc0
    filter: fz
    dx: nc1hwc0 -> nchw
    """
    dtype = "float16"

    with OpContext("dynamic"):
        with operation.ComputeContext():
            y = {
                'shape': [-1, -1, -1, -1, -1],
                'ori_format': "NCHW",
                'ori_shape': [-1, -1, -1, -1],
                'dtype': dtype,
                'range': [(1, None)] * 4
            }

            var_filter_ci1hw = operation.var("filter_ci1hw")
            var_filter_co1 = operation.var("filter_co1")
            var_n = operation.var("batch_n")
            var_dedy_c1 = operation.var("dedy_c1")
            var_dedy_h = operation.var("dedy_h")
            var_dedy_w = operation.var("dedy_w")
            dy_shape_nc1hwc0 = (var_n, var_dedy_c1, var_dedy_h, var_dedy_w, 16)
            filter_fz = (var_filter_ci1hw, var_filter_co1, 16, 16)

            input_tensor = tvm.placeholder([4], name="input_size", dtype="int32")
            dy_tensor = tvm.placeholder(dy_shape_nc1hwc0, name="dedy", dtype=dtype)
            filter_tensor = tvm.placeholder(filter_fz, name="filter", dtype=dtype)

            tensor_list = [input_tensor, filter_tensor, dy_tensor]
            _build_dx_transdata_fusion_op(tensor_list, y, (1, 1, 2, 2), dy_tensor, "dx_transdata_fusion_binary")


def _test_transdata_dx_transdata_fusion_op(test_args):
    """
    transdata + dx + transdata
    dy: nchw -> nc1hwc0
    filter: fz
    dx: nc1hwc0 -> nchw
    """
    dtype = "float16"
    with OpContext("dynamic"):
        with operation.ComputeContext():
            y = {
                'shape': [-1, -1, -1, -1, -1],
                'ori_format': "NCHW",
                'ori_shape': [-1, -1, -1, -1],
                'dtype': dtype,
                'range': [(1, None)] * 4
            }

            var_n = operation.var("batch_n")
            var_dedy_c= operation.var("dedy_c")
            var_dedy_h = operation.var("dedy_h")
            var_dedy_w = operation.var("dedy_w")
            var_filter_ci1hw = operation.var("filter_ci1hw")
            var_filter_co1 = operation.var("filter_co1")
            dy_shape_nchw = (var_n, var_dedy_c, var_dedy_h, var_dedy_w)
            filter_fz = (var_filter_ci1hw, var_filter_co1, 16, 16)

            transdata_in_tensor = tvm.placeholder(dy_shape_nchw, name="transdata_in", dtype=dtype)
            dy_tensor = trans_data_fusion_compute(transdata_in_tensor, {}, "NCHW", "NC1HWC0")
            input_tensor = tvm.placeholder([4], name="input_size", dtype="int32")
            filter_tensor = tvm.placeholder(filter_fz, name="filter", dtype=dtype)
            tensor_list = [input_tensor, filter_tensor, transdata_in_tensor]
            _build_dx_transdata_fusion_op(tensor_list, y, (1, 1, 1, 1), dy_tensor, "transdata_dx_transdata_fusion_binary")


def _test_transdata_dx_transdata_shape_util(test_arg):
    inputs = [
        {"ori_shape": [4], "shape": [4], "data_type": "int32",
         "ori_range": [[4, 4]], "range": [[4, 4]],
         "format": "NCHW", "ori_format": "NCHW", "input_pattern": "cube", "input_op_type": "Conv2DBackpropInput"},
        {"ori_shape": [3, 3, 16, 16], "shape": [9, 1, 16, 16], "data_type": "float16",
         "ori_range": [[3, 3], [3, 3], [16, 16], [16, 16]], "range": [[9, 9], [1, 1], [16, 16], [16, 16]],
         "format": "FRACTAL_Z", "ori_format": "HWCN", "input_pattern": "cube", "input_op_type": "Conv2DBackpropInput"},
    ]
    shape_util.variable_shape(inputs, op_mode="cube")


def _test_dx_transdata_shape_util(test_arg):
    inputs = [
        {"ori_shape": [4], "shape": [4], "data_type": "int32",
         "ori_range": [[4, 4]], "range": [[4, 4]],
         "format": "NCHW", "ori_format": "NCHW", "input_pattern": "cube", "input_op_type": "Conv2DBackpropInput"},
        {"ori_shape": [3, 3, 32, 64], "shape": [18, 4, 16, 16], "data_type": "float16",
         "ori_range": [[3, 3], [3, 3], [32, 32], [64, 64]], "range": [[18, 18], [4, 4], [16, 16], [16, 16]],
         "format": "FRACTAL_Z", "ori_format": "HWCN", "input_pattern": "cube", "input_op_type": "Conv2DBackpropInput"},
        {"ori_shape": [-1, -1, -1, -1], "shape": [-1, -1, -1, -1, 16], "data_type": "float16",
         "ori_range": [[1, -1], [1, -1], [1, -1], [1, -1]], "range": [[1, None], [1, None], [1, None], [1, None], [16, 16]],
         "format": "NC1HWC0", "ori_format": "NCHW", "input_pattern": "cube", "input_op_type": "Conv2DBackpropInput"},
    ]
    shape_util.variable_shape(inputs, op_mode="cube")

ut_case.add_cust_test_func(support_soc="Ascend910A", test_func=_test_dx_transdata_fusion_op)
ut_case.add_cust_test_func(support_soc="Ascend910A", test_func=_test_transdata_dx_transdata_fusion_op)
ut_case.add_cust_test_func(support_soc="Ascend910A", test_func=_test_transdata_dx_transdata_shape_util)
ut_case.add_cust_test_func(support_soc="Ascend910A", test_func=_test_dx_transdata_shape_util)


if __name__ == '__main__':
    ut_case.run("Ascend910A")