#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3D
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3D", "impl.dynamic.conv3d", "conv3d")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api_end_with_d(
    fmap={'ori_shape': (1, -1, -1, -1, 32), 'shape': (1, -1, -1, -1, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16',
          "range": [(1, 1), (4, 18), (50, 70), (78, 98), (32,32)]},
    weight={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16',"range": [(2, 2), (2, 2), (2, 2), (32, 32), (64,64)]},
    bias=None, offset_w=None,
    output={'ori_shape': (1, 4, 30, 44, 64), 'shape': (1, 4, 30, 44, 64),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (4, 18), (50, 70), (78, 98), (32,32)]},
    strides=(1, 2, 2, 2, 1),
    pads=[0, 0, 0, 0, 0, 0],
    dilations=(1, 1, 1, 1, 1),
    groups=1, data_format="NDHWC", offset_x=0):
    return [fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x]


def test_conv3d_fuzzy_build_generalization(case):
    from impl.dynamic.conv3d import conv3d_generalization
    input_list = case.get("inputs")
    expect = case.get("expect")
    case_name = input_list[-2]
    def _test_generalization_function(test_arg):
        res = conv3d_generalization(*input_list)
        if expect == "success":
            if not res[0][0].get("ori_range"):
                raise RuntimeError(f"In case {case_name}, conv3d_generalization function expected to \
                    generate ori_range success.")
        elif expect == "unsupported":
            if res[0].get("result") != "UNSUPPORTED":
                raise RuntimeError(f"In case {case_name}, conv3d_generalization function expected to return {expect}.")
        elif expect not in res[0].get("reason").get("type"):
            raise RuntimeError(f"In case {case_name}, conv3d_generalization function expected to return {expect}.")
    return _test_generalization_function


def _test_op_get_op_support_info(test_arg):
    from impl.dynamic.conv3d import get_op_support_info
    [fmap, weight, bias, offset_w, output, strides,
     pads, dilations, groups, data_format, offset_x] = _run_api_end_with_d()
    fmap['format'] = 'NDC1HWC0'
    get_op_support_info(
        fmap, weight, bias, offset_w, output, strides,
        pads, dilations, groups, data_format, offset_x)
    # Test Else branch
    fmap['format'] = 'NDHWC'
    get_op_support_info(
        fmap, weight, bias, offset_w, output, strides,
        pads, dilations, groups, data_format, offset_x)


def _test_op_get_op_support_info_bias(test_arg):
    from impl.dynamic.conv3d import get_op_support_info
    [fmap, weight, bias, offset_w, output, strides,
     pads, dilations, groups, data_format, offset_x] = _run_api_end_with_d()
    fmap['format'] = 'NDC1HWC0'
    bias = {'ori_shape': (32,), 'shape': (32,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    get_op_support_info(
        fmap, weight, bias, offset_w, output, strides,
        pads, dilations, groups, data_format, offset_x)


def _test_op_get_op_support_info_wrong_format(test_arg):
    from impl.dynamic.conv3d import get_op_support_info
    [fmap, weight, bias, offset_w, output, strides,
     pads, dilations, groups, data_format, offset_x] = _run_api_end_with_d()
    fmap['format'] = 'NDC1HWC0'
    # wrong filter
    wrong_filter = weight.copy()
    wrong_filter['ori_format'] = 'NDDDD'
    wrong_filter['format'] = 'NDDDD'
    try:
        get_op_support_info(
            fmap, wrong_filter, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x)
    except Exception as e:
        print(e)
    # wrong strides format
    wrong_fmap = fmap.copy()
    wrong_fmap['ori_format'] = 'NDDDD'
    wrong_fmap['format'] = 'NDDDD'
    try:
        get_op_support_info(
            wrong_fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x)
    except Exception as e:
        print(e)


# test_conv3d_fuzzy_build_generalization
print("adding conv3d test_conv3d_fuzzy_build_generalization testcase")
fuzzy_test_case = [
    {"inputs": [
        {'shape': (2, 8, 8, 8, 320),
         'ori_shape': (2, 8, 8, 8, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (2, 4, 4, 4, 320),
         'ori_shape': (2, 4, 4, 4, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'}, (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_static_mode_general_case', {"mode": "keep_rank"}],
     "expect": "success"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (4, 10), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (4, 10), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_general_case', {"mode": "keep_rank"}],
     "expect": "success"},
    {"inputs": [
        {'shape': (2, 8, 8, 8, 320),
         'ori_shape': (2, 320, 8, 8, 8),
         'ori_format': 'NCDHW',
         'format': 'NDHWC',
         'dtype': 'float16'},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (2, 4, 4, 4, 320),
         'ori_shape': (2, 320, 4, 4, 4),
         'ori_format': 'NCDHW',
         'format': 'NDHWC',
         'dtype': 'float16'}, (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_static_mode_format_case', {"mode": "keep_rank"}],
     "expect": "success"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, 320, -1, -1, -1),
         'ori_format': 'NCDHW',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (320, 320), (2, 10), (2, 10), (4, 10)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, 320, -1, -1, -1),
         'ori_format': 'NCDHW',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (320, 320), (2, 10), (2, 10), (4, 10)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_format_case', {"mode": "keep_rank"}],
     "expect": "success"},
    {"inputs": [
        {'shape': (-2,),
         'ori_shape': (-2,),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(1, None), (1, None), (1, None), (1, None), (1, None)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (1, 10), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_rank_case', {"mode": "keep_rank"}],
     "expect": "unsupported"},
    {"inputs": [
        {'shape': (2, 8, 5000, 8, 320),
         'ori_shape': (2, 8, 5000, 8, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (2, 4, 2500, 4, 320),
         'ori_shape': (2, 4, 2500, 4, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'}, (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_static_mode_h_dim_large_case', {"mode": "keep_rank"}],
     "expect": "unsupported"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 5000), (4, 10), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 5000), (4, 10), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_h_dim_upper_range_large_case', {"mode": "keep_rank"}],
     "expect": "upper_limit"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (5000, 5000), (4, 10), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (5000, 5000), (4, 10), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_h_dim_lower_range_large_case', {"mode": "keep_rank"}],
     "expect": "lower_limit"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (2, 10), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (1, 10), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_w_lower_case', {"mode": "keep_rank"}],
     "expect": "lower_limit"},
    {"inputs": [
        {'shape': (2, 8, 8, 8, 320),
         'ori_shape': (2, 8, 8, 8, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 500, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (2, 4, 4, 4, 320),
         'ori_shape': (2, 4, 4, 4, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'}, (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_static_mode_invalid_weight_case', {"mode": "keep_rank"}],
     "expect": "unsupported"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (4, 10), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 2, 2, 500, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(2, 10), (2, 10), (2, 10), (4, 10), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_invalid_weight_case', {"mode": "keep_rank"}],
     "expect": "unsupported"},
    {"inputs": [
        {'shape': (2, 8, 8, 4095, 320),
         'ori_shape': (2, 8, 8, 4095, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 8, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (2, 4, 4, 2048, 320),
         'ori_shape': (2, 4, 4, 2048, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16'}, (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_static_mode_exceed_l1_case', {"mode": "keep_rank"}],
     "expect": "unsupported"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(8, 10), (8, 10), (8, 10), (8, 4095), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 8, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(8, 10), (8, 10), (8, 10), (8, 2048), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_upper_range_exceed_l1_case', {"mode": "keep_rank"}],
     "expect": "upper_limit"},
    {"inputs": [
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(8, 10), (8, 10), (8, 10), (4095, 4095), (320, 320)]},
        {'shape': (160, 20, 16, 16),
         'ori_shape': (2, 8, 2, 320, 320),
         'ori_format': 'DHWCN',
         'format': 'FRACTAL_Z_3D',
         'dtype': 'float16'}, None, None,
        {'shape': (-1, -1, -1, -1, 320),
         'ori_shape': (-1, -1, -1, -1, 320),
         'ori_format': 'NDHWC',
         'format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(8, 10), (8, 10), (8, 10), (2048, 2048), (320, 320)]},
        (1, 2, 2, 2, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 0,
        'test_conv3d_generalization_dynamic_mode_lower_range_exceed_l1_case', {"mode": "keep_rank"}],
     "expect": "lower_limit"},
     {"inputs": [
        {'ori_shape': (1, 44, 24, 8, 8),
         'ori_format': 'NCDHW',
         'dtype': 'float16'},
        {'ori_shape': (112, 5, 1, 7, 44),
         'ori_format': 'NDHWC',
         'dtype': 'float16'}, None, None,
        {'ori_shape': (1, 112, 8, 3, 3),
         'ori_format': 'NCDHW',
         'dtype': 'float16'}, (1, 1, 3, 3, 3), (1, 1, 0, 0, 2, 3), (1, 1, 1, 1, 1), 1, 'NCDHW', 0,
        'test_conv3d_generalization_static_mode_w_unsupport_case', {"mode": "keep_rank"}],
     "expect": "unsupported"},
     {"inputs": [
        {'ori_shape': (1, 44, 24, 8, 10),
         'ori_format': 'NCDHW',
         'dtype': 'float16'},
        {'ori_shape': (112, 5, 1, 7, 44),
         'ori_format': 'NDHWC',
         'dtype': 'float16'}, None, None,
        {'ori_shape': (1, 112, 8, 3, 3),
         'ori_format': 'NCDHW',
         'dtype': 'float16'}, (1, 1, 3, 3, 3), (1, 1, 0, 0, 2, 3), (1, 1, 1, 1, 1), 1, 'NCDHW', 0,
        'test_conv3d_generalization_static_mode_w_range_modify_case', {"mode": "keep_rank"}],
     "expect": "success"}
]
for case in fuzzy_test_case:
    ut_case.add_cust_test_func('Ascend910A', test_func=test_conv3d_fuzzy_build_generalization(case))

ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)
ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info_bias)
ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info_wrong_format)

# test_conv3dbp_succ_d
case1 = _run_api_end_with_d()

# test_conv3dbp_stride_one
fmap = {'ori_shape': (-1, 32, 8, 60, 88), 'shape': (-1, 32, 8, 60, 88),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (8, 8), (60, 60), (88,88)]}
weight = {'ori_shape': (64, 32, 2, 2, 2), 'shape': (64, 32, 2, 2, 2),
          'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16',"range": [(64, 64), (32, 32), (2, 2), (2, 2), (2,2)]}
output = {'ori_shape': (1, 7, 59, 87, 64), 'shape': (1, 7, 59, 87, 64),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 10), (32, 32), (8, 8), (60, 60), (88,88)]}
strides = (1, 1, 1, 1, 1)
data_format="NCDHW"
case2 = _run_api_end_with_d(fmap=fmap, weight=weight, output=output, strides=strides, data_format=data_format)

# test_bias_length_fail
bias = {'ori_shape': (64, 64), 'shape': (64, 64),
        'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
case3 = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (-1, 32, 15, 4098, 18), 'shape': (-1, 32, 15, 4098, 18),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (15, 15), (4098, 4098), (18,18)]}
case4 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_output
output = {'dtype': 'float32'}
case5 = _run_api_end_with_d(output=output)

# test_conv3d_invalid_dilations
dilations = (1, 2, 1, 1, 1)
case6 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (1, 8, 60, 88), 'shape': (1, 8, 60, 88),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 10), (8, 8), (60, 60), (88, 88)]}
case7 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_pad_length
pads = (0, -1, -1, -1, 0)
case8 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 2, 32), 'shape': (2, 2, 2, 32),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case9 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight_D
weight = {'ori_shape': (2, 2, 354, 32, 64), 'shape': (2, 2, 354, 32, 64),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case10 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_big_fmap
fmap = {'ori_shape': (-1, 3000, 4000, 4000, 3000),
        'shape': (-1, 3000, 4000, 4000, 3000),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 200), (3000, 3000), (4000, 4000), (4000, 4000), (3000,3000)]}
case11 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_bias_dtype
bias = {'ori_shape': (1,), "dtype": "float32"}
case12 = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_pads_w
pads = (1, 1, 1, 1, 3, 1)
case13 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 0, 1, 1)
case14 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_fmap_format
fmap = {'ori_shape': (-1, 32, 8, 60, 88), 'shape': (1, 32, 8, 60, 88),
        'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16', "range": [(1, 10), (32, 32), (8, 8), (60, 60), (88,88)]}
case15 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
          'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16'}
case16 = _run_api_end_with_d(weight=weight)

# test_conv3d_load2d_dhw
fmap = {'ori_shape': (1, 15, -1, -1, -1), 'shape': (1, 15, -1, -1, -1),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16', "range": [(1, 1), (15, 15), (10, 20), (10, 20), (10, 20)]}
weight = {'ori_shape': (32, 15, 1, 1, 1), 'shape': (32, 15, 1, 1, 1),
          'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16',"range": [(32, 32), (15, 15), (1, 1), (1, 1), (1,1)]}
output = {'ori_shape': (1, 32, -1, -1, -1), 'shape': (1, 32, -1, -1, -1),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 20), (10, 20), (10, 20)]}
strides = (1, 1, 1, 1, 1)
pads=[-1, -1, -1, -1, -1, -1]
data_format="NCDHW"
case17 = _run_api_end_with_d(fmap=fmap, weight=weight, output=output, strides=strides, pads=pads, data_format=data_format)

# test_conv3d_stride_h_opti_dhw
fmap = {'ori_shape': (1, 15, -1, -1, -1), 'shape': (1, 15, -1, -1, -1),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16', "range": [(1, 1), (15, 15), (10, 20), (10, 20), (10, 20)]}
weight = {'ori_shape': (32, 15, 1, 1, 1), 'shape': (32, 15, 1, 1, 1),
          'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16',"range": [(32, 32), (15, 15), (1, 1), (1, 1), (1,1)]}
output = {'ori_shape': (1, 32, -1, -1, -1), 'shape': (1, 32, -1, -1, -1),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 20), (10, 20), (10, 20)]}
strides = (1, 1, 1, 2, 1)
pads=[-1, -1, -1, -1, -1, -1]
data_format = "NCDHW"
case18 = _run_api_end_with_d(fmap=fmap, weight=weight, output=output, strides=strides, pads=pads, data_format=data_format)

# test_conv3d_invalid_stride_shape
strides = (1, 0, 1, 1, 1)
case19 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 0, 1, 1)
case20 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 1, 0, 1)
case21 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_weight
weight = {'ori_shape': (257, 2, 2, 32, 64), 'shape': (257, 2, 2, 32, 64),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case22 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 257, 2, 32, 64), 'shape': (2, 257, 2, 32, 64),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case23 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_dilations
dilations = (1, 0, 1, 1, 1)
case24 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_pad_length
pads = (256, 256, 256, 256, 256, 256)
case25 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_fmap_shape
pads = (2, 2, 256, 256, 256, 256)
case26 = _run_api_end_with_d(pads=pads)

# test_conv3d_with_bias
bias = {'ori_shape': (64,), 'shape': (64,),
        'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16', 'range':[(64, 64),]}
case27 = _run_api_end_with_d(bias=bias)

# test_conv3d_with_bias and double buffer (UB reused success)
fmap = {'ori_shape': (-1, 4, 224, -1, 3), 'shape': (-1, 4, 1, 224, -1, 16),
        'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16',
        "range": [(1, 66), (4, 4), (1, 1), (224, 224),(224, 537), (16, 16)]}
weight = {'ori_shape': (4, 7, 7, 3, 64), 'shape': (4, 7, 7, 3, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
bias = {'ori_shape': (64,), 'shape': (64,),
        'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16', 'range':[(64, 64),]}
strides = (1, 2, 2, 2, 1)
pads = [-1, -1, -1, -1, -1, -1]
case28 = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides, pads=pads)

# test y_w_lower smaller than 2 in SAME PADDING
fmap={'ori_shape': (1, -1, -1, -1, 32), 'shape': (1, -1, -1, -1, 32),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (2, 18), (50, 70), (1, 10), (32,32)]}
pads = [-1, -1 ,-1, -1, -1, -1]
case29 = _run_api_end_with_d(fmap=fmap,pads=pads)

# test y_d_lower smaller than 2
# test y_h_lower smaller than 2
# test y_w_lower smaller than 2
fmap={'ori_shape': (1, -1, -1, -1, 32), 'shape': (1, -1, -1, -1, 32),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (1, 18), (1, 10), (1, 10), (32,32)]}
case30 = _run_api_end_with_d(fmap=fmap)

# test fmap_h + padding < filter_h constraint
fmap = {'ori_shape': (1, -1, 1, -1, 32), 'shape': (1, -1, 1, -1, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (2, 18), (1, 1), (78, 98), (32,32)]}
case31 = _run_api_end_with_d(fmap=fmap)

# Test padding H invalid constraint
fmap = {'ori_shape': (1, -1, -1, -1, 32), 'shape': (1, -1, -1, -1, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (2, 18), (50, 70), (78, 98), (32,32)]}
invalid_pads = [0, 0, 270, 270, 0, 0]
case32 = _run_api_end_with_d(fmap=fmap,pads=invalid_pads)

# test NO_UPPER_LIMIT range in No pad Mode
fmap={'ori_shape': (1, -1, -1, -1, 32), 'shape': (1, -1, -1, -1, 32),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (4000, None), (4000, None), (4000, None), (32,32)]}
case33 = _run_api_end_with_d(fmap=fmap)

# test NO_UPPER_LIMIT range in pad same Mode
fmap={'ori_shape': (1, -1, -1, -1, 32), 'shape': (1, -1, -1, -1, 32),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (4000, None), (4000, None), (4000, None), (32,32)]}
pads = [-1, -1, -1, -1, -1, -1]
case34 = _run_api_end_with_d(fmap=fmap,pads=pads)

# Test padding w invalid constraint
fmap = {'ori_shape': (1, -1, -1, -1, 32), 'shape': (1, -1, -1, -1, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (2, 18), (50, 70), (78, 98), (32,32)]}
invalid_pads = [0, 0, 0, 0, 999, 999]
case35 = _run_api_end_with_d(fmap=fmap,pads=invalid_pads)

# test_conv3d_invalid_pads h dim
pads = (1, 1, 3, 3, 1, 1)
case36 = _run_api_end_with_d(pads=pads)

# test fmap_w + padding < filter_w constraint
fmap = {'ori_shape': (1, -1, -1, 1, 32), 'shape': (1, -1, -1, 1, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, 1), (2, 18), (50, 70), (1, 1), (32,32)]}
case37 = _run_api_end_with_d(fmap=fmap)

# test fmap = [-2]
fmap = {'ori_shape': (-2,), 'shape': (-2,),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
case38 = _run_api_end_with_d(fmap=fmap)
# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case1, "success", "dynamic_case1", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                _gen_data_case(case2, "success", "dynamic_case2", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case3, RuntimeError, "dynamic_case3", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case4, RuntimeError, "dynamic_case4", True))

# ut_case.add_case(["Ascend910A", "Ascend310"],
#                  _gen_data_case(case5, RuntimeError, "dynamic_case5", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case6, RuntimeError, "dynamic_case6", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case7, RuntimeError, "dynamic_case7", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case8, RuntimeError, "dynamic_case8", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case9, RuntimeError, "dynamic_case9", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case10, RuntimeError, "dynamic_case10", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case11, RuntimeError, "dynamic_case11", True))

# ut_case.add_case(["Ascend910A", "Ascend310"],
#                  _gen_data_case(case12, RuntimeError, "dynamic_case12", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case13, RuntimeError, "dynamic_case13", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case14, RuntimeError, "dynamic_case14", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case15, RuntimeError, "dynamic_case15", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case16, RuntimeError, "dynamic_case16", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case17, "success", "dynamic_case17", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case18, "success", "dynamic_case18", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case19, RuntimeError, "dynamic_case19", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case20, RuntimeError, "dynamic_case20", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case21, RuntimeError, "dynamic_case21", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case22, RuntimeError, "dynamic_case22", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case23, RuntimeError, "dynamic_case23", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case24, RuntimeError, "dynamic_case24", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case25, RuntimeError, "dynamic_case25", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case26, RuntimeError, "dynamic_case26", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case27, "success", "dynamic_case27", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case28, "success", "dynamic_case28_test_bias_reused", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case29, "success", "dynamic_case29", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case30, "success", "dynamic_case30", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case31, RuntimeError, "dynamic_case31", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case32, RuntimeError, "dynamic_case32", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case33, "success", "dynamic_case33", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case34, "success", "dynamic_case34", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case35, RuntimeError, "dynamic_case35", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case36, RuntimeError, "dynamic_case36", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case37, RuntimeError, "dynamic_case37", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case38, RuntimeError, "dynamic_case38", True))

# ut for tilingcase fuzzy compile
def test_conv3d_fuzz_build_tilingcase(test_arg):
    import json
    from impl.dynamic.conv3d import conv3d
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
                                        [16, 32],
                                        [32, 32],
                                        [1, 2],
                                        [16, 32],
                                        [16, 32]
                                    ],
                                    "shape": [-1, 32, -1, -1, -1]
                                }]
                            }]
                        }]
        get_context().add_addition("missing_support_info", json.dumps(missing_info))

        input_list = [
            {
                'shape': (-1, -1, 2, -1, -1, 16),
                'ori_shape': (-1, 32, -1, -1, -1),
                'ori_format': 'NCDHW',
                'format': 'NDC1HWC0',
                'dtype': 'float16',
                'range': ((16, 32), (1, 2), (2, 2), (16, 32), (16, 32), (16, 16))
            }, {
                'ori_shape': (32, 32, 1, 1, 1),
                'shape': (2, 2, 16, 16),
                'ori_format': 'NCDHW',
                'format': 'FRACTAL_Z_3D',
                'dtype': 'float16'
            }, None, None, {
                'shape': (-1, -1, 2, -1, -1, 16),
                'ori_shape': (-1, 32, -1, -1, -1),
                'ori_format': 'NCDHW',
                'format': 'NDC1HWC0',
                'dtype': 'float16',
                'range':  ((16, 32), (1, 2), (2, 2), (16, 32), (16, 32), (16, 16))
            }, (1, 1, 1, 1, 1), [0, 0, 0, 0, 0, 0], (1, 1, 1, 1, 1), 1, 'NCDHW', 0, 'conv3d_fuzz_build_generalization']
        conv3d(*input_list)
print("adding test_conv3d_fuzz_build_tilingcase testcase")
ut_case.add_cust_test_func(support_soc=('Ascend910A'), test_func=test_conv3d_fuzz_build_tilingcase)

if __name__ == '__main__':
    ut_case.run("Ascend910A")