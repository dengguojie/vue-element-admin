#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInputD
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropInputD", "impl.conv3d_backprop_input_d",
               "conv3d_backprop_input_d")


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api_end_with_d(
    filters={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
             'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'},
    out_backprop={'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
                  'ori_format': 'NDHWC', 'format': 'NDHWC',
                  'dtype': 'float16'},
    y_input={'ori_shape': (1, 16, 120, 176, 32),
             'shape': (1, 16, 120, 176, 32),
             'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    input_sizes=(1, 16, 120, 176, 32),
    strides=(1, 2, 2, 2, 1),
    pads=[0, 0, 0, 0, 0, 0],
    dilations=(1, 1, 1, 1, 1),
    groups=1, data_format="NDHWC"):
    return [filters, out_backprop, y_input, input_sizes, strides,
            pads, dilations, groups, data_format]


def test_op_check_supported(test_arg):
    from impl.conv3d_backprop_input_d import check_supported
    input_sizes = (1, 16, 120, 176, 16)
    (filters, out_backprop, y_input, input_sizes, strides,
                    pads, dilations, groups, data_format) = _run_api_end_with_d(input_sizes=input_sizes)
    check_supported(filters, out_backprop, y_input, input_sizes, strides,
                                pads, dilations, groups, data_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

def _test_op_get_op_support_info(test_arg):
    from impl.conv3d_backprop_input_d import get_op_support_info

    [filters, out_backprop, y_input, input_sizes, strides,
     pads, dilations, groups, data_format] = _run_api_end_with_d()
    out_backprop["format"] = "NDC1HWC0"
    get_op_support_info(
        filters, out_backprop, y_input, input_sizes, strides,
        pads, dilations, groups, data_format)

    # test wrong out_backprop format
    wrong_out_backprop = {'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
                          'ori_format': 'NNNNN', 'format': 'NNNNN',
                          'dtype': 'float16'}
    try:
        get_op_support_info(
            filters, wrong_out_backprop, y_input, input_sizes, strides,
            pads, dilations, groups, data_format)
    except Exception as e:
        print(e)

    # test wrong filter format
    wrong_filters = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
                    'ori_format': 'NNNNN', 'format': 'NNNNN', 'dtype': 'float16'}
    try:
        get_op_support_info(
            wrong_filters, out_backprop, y_input, input_sizes, strides,
            pads, dilations, groups, data_format)
    except Exception as e:
        print(e)

    # test wrong y format
    wrong_res = {'ori_shape': (1, 16, 120, 176, 32),
                 'shape': (1, 16, 120, 176, 32),
                 'ori_format': 'NNNNN', 'format': 'NNNNN', 'dtype': 'float16'}
    try:
        get_op_support_info(
            filters, out_backprop, wrong_res, input_sizes, strides,
            pads, dilations, groups, data_format)
    except Exception as e:
        print(e)

    # test cal l1_1 size
    y_input = {'ori_shape': (1, 16, 120, 2, 32), 'shape': (1, 16, 120, 2, 32),
               'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    try:
        get_op_support_info(
            filters, out_backprop, y_input, input_sizes, strides,
            pads, dilations, groups, data_format)
    except Exception as e:
        print(e)

ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)

# test_conv3dbp_succ_d
case1 = _run_api_end_with_d()


# test_conv3dbp_NCDHW
filters = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (1, 64, 8, 60, 88), 'shape': (1, 64, 8, 60, 88),
                'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
y_input = {'ori_shape': (1, 32, 16, 120, 176), 'shape': (1, 32, 16, 120, 176),
           'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
input_sizes = (1, 32, 16, 120, 176)
strides = (1, 1, 2, 2, 2)
case2 = _run_api_end_with_d(
    filters=filters, out_backprop=out_backprop, y_input=y_input,
    input_sizes=input_sizes, strides=strides, data_format='NCDHW')

# test_conv3dbp_invalid_shape
filters = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
           'ori_format': 'DHCWN', 'format': 'DHCWN', 'dtype': 'float16'}
case3 = _run_api_end_with_d(filters=filters)

# test_conv3dbp_outbackprop_wrong_format
out_backprop = {'ori_shape': (8, 60, 88, 64, 1), 'shape': (8, 60, 88, 64, 1),
                'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case4 = _run_api_end_with_d(out_backprop=out_backprop)

# Invalid pads
pads = [0, 0, 0, 0]
case5 = _run_api_end_with_d(pads=pads)

# test_conv3dbp_invalid_dilations
dilations = [1, 0, 1, 0]
case6 = _run_api_end_with_d(dilations=dilations)

# test_conv3dbp_invalid_shape
# fmap_channel != filter_channel
input_sizes = (1, 16, 120, 176, 16)
case7 = _run_api_end_with_d(input_sizes=input_sizes)

# dedy_channel != filter_batch
filters = {'ori_shape': (2, 2, 2, 32, 32), 'shape': (2, 2, 2, 32, 32),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case8 = _run_api_end_with_d(filters=filters)

# fmap_batch != dedy_batch
input_sizes = (2, 16, 120, 176, 32)
case9 = _run_api_end_with_d(input_sizes=input_sizes)

# Filter with NDHWC but failed.
filters = {'ori_shape': (64, 2, 2, 2, 32), 'shape': (64, 2, 2, 2, 32),
           'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case10 = _run_api_end_with_d(filters=filters)

# Wrong out_backprop shape
out_backprop = {'ori_shape': (8, 6, 8, 6, 1), 'shape': (8, 6, 8, 6, 1),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case11 = _run_api_end_with_d(out_backprop=out_backprop)

# Wrong dataformat.
data_format = "DHWNC"
case12 = _run_api_end_with_d(data_format=data_format)

# Wrong filter data format
filters = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
           'ori_format': 'DNHCW', 'format': 'DNHCW', 'dtype': 'float16'}
case13 = _run_api_end_with_d(filters=filters)

# Wrong y data format
y_input = {'ori_shape': (1, 16, 120, 176, 32), 'shape': (1, 16, 120, 176, 32),
           'ori_format': 'DNHCW', 'format': 'DNHCW', 'dtype': 'float16'}
case14 = _run_api_end_with_d(y_input=y_input)

# Wrong dilations
dilations = [2, 1, 1, 1, 2]
case15 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_pads_dtype
pads = {"2": 2}
case16 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_input_sizes
input_sizes = (1, 35, 6, 12, 176)
case17 = _run_api_end_with_d(input_sizes=input_sizes)

# test_filter_NCDHW
filters = {'ori_shape': (64, 2, 2, 2, 32), 'shape': (64, 2, 2, 2, 32),
           'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
case18 = _run_api_end_with_d(filters=filters)

# test_filter_NCDHWC FAIL
filters = {'ori_shape': (64, 2, 2, 2, 32), 'shape': (64, 2, 2, 2, 32),
           'ori_format': 'NCDHWC', 'format': 'NCDHWC', 'dtype': 'float16'}
case19 = _run_api_end_with_d(filters=filters)

# wrong filter_d_dilation > fmap_d_padding
filters = {'ori_shape': (17, 2, 2, 32, 64), 'shape': (17, 2, 2, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case20 = _run_api_end_with_d(filters=filters)

# wrong out_d size
strides = (1, 4, 2, 2, 1)
case21 = _run_api_end_with_d(strides=strides)

# wrong filter_h_dilation > fmap_h_padding
filters = {'ori_shape': (1, 20, 1, 32, 64), 'shape': (1, 20, 1, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (1, 2, 4, 10, 64), 'shape': (1, 2, 4, 10, 64),
              'ori_format': 'NDHWC', 'format': 'NDHWC',
              'dtype': 'float16'}
input_sizes = (1, 2, 40, 4, 64)
case22 = _run_api_end_with_d(out_backprop=out_backprop, input_sizes=input_sizes, filters=filters)

# wrong out_h size
strides = (1, 2, 10, 2, 1)
case23 = _run_api_end_with_d(strides=strides)

# wrong filter_w_dilation > fmap_w_padding
filters = {'ori_shape': (1, 1, 20, 32, 64), 'shape': (1, 1, 20, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (1, 2, 40, 10, 64), 'shape': (1, 2, 40, 10, 64),
              'ori_format': 'NDHWC', 'format': 'NDHWC',
              'dtype': 'float16'}
input_sizes = (1, 2, 40, 10, 64)
case24 = _run_api_end_with_d(out_backprop=out_backprop, input_sizes=input_sizes, filters=filters)

# wrong out_h size
strides = (1, 2, 2, 3, 1)
case25 = _run_api_end_with_d(strides=strides)

# dilation_d != 1
dilations = (1, 2, 1, 1, 1)
case26 = _run_api_end_with_d(dilations=dilations)

# exceed l1_1
filters = {'ori_shape': (128, 1, 1, 32, 64), 'shape': (128, 1, 1, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (1, 8, 4096, 16, 64), 'shape': (1, 8, 4096, 16, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC',
                'dtype': 'float16'}
input_sizes = (1, 128, 4096, 16, 32)
strides = (1, 1, 1, 1, 1)
case27 = _run_api_end_with_d(out_backprop=out_backprop, filters=filters,
                             input_sizes=input_sizes, strides=strides)

# exceed _check_attr_range
strides = (1, 2, 2, 3000, 1)
case28 = _run_api_end_with_d(strides=strides)

# test H!=1 W=1
case_wo1 = _run_api_end_with_d(
    filters={'ori_shape': (1, 5, 8, 52, 11), 'shape': (1, 5, 8, 52, 11),
             'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'},
    out_backprop={'ori_shape': (1, 3, 28, 1, 11), 'shape': (1, 3, 28, 1, 11),
                  'ori_format': 'NDHWC', 'format': 'NDHWC',
                  'dtype': 'float16'},
    y_input={'ori_shape': (1, 52, 28, 1, 52),
             'shape': (1, 52, 28, 1, 52),
             'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    input_sizes=(1, 52, 28, 1, 52),
    strides=(1, 20, 1, 4, 1),
    pads=[0, 0, 2, 2, 3, 4],
    dilations=(1, 1, 1, 1, 1),
    groups=1, data_format="NDHWC")

# default tiling
def test_conv3d_dx_mock_default_tiling(test_args):
    from tbe.common.tiling.tiling_helper import TILING_INSTANCE
    from impl.conv3d_backprop_input_d import conv3d_backprop_input_d
    tiling_type = "auto_tiling"
    tiling_params = {'a_shape': [1, 1, 1, 1, 1, 16], 'b_shape': [16, 2, 1, 2, 2, 16], 'c_shape': [1, 2, 2, 2, 16],
                     'a_dtype': 'float16', 'b_dtype': 'float16', 'c_dtype': 'float16', 'mad_dtype': 'float32',
                     'pad': [0, 0, 1, 0, 1, 0], 'stride': [2, 1, 1], 'strideh_expand': 2, 'stridew_expand': 2,
                     'dilation': [1, 1, 1], 'group': 1, 'fused_coefficient': [0, 0, 0], 'bias_flag': False,
                     'op_type': 'conv3d_backprop_input', 'kernel_name': 'conv3d_dx_fault',
                     'model_type': 'xgboost', 'dynamic_shape_flag': False, 'fused_channel_wise': [0, 0, 0],
                     'fusion_type': 0, 'l1_fusion_type': -1, 'l2_fusion_type': -1, 'fm_l1_valid_size': 0,
                     'fm_l1_valid_size_level': 0}
    tiling_dict = {
        'conv3d_dx_fault': {'AL0_matrix': [50, 1, 32, 16, 1, 2], 'AL1_shape': [224, 1, 1, 1],
                            'AUB_channel_wise_flag': None, 'AUB_shape': [896, 3, 1, 1], 'A_overhead_opt_flag': 0,
                            'BL0_matrix': [1, 2, 16, 16, 1, 1], 'BL1_shape': [], 'BUB_channel_wise_flag': None,
                            'BUB_shape': [1, 1, 1, 1], 'B_overhead_opt_flag': 0, 'CL0_matrix': [2, 50, 16, 16, 1, 1],
                            'CUB_channel_wise_flag': False, 'CUB_matrix': [1, 50, 16, 16, 1, 1],
                            'batch_bef_group_flag': 0, 'block_dim': [1, 3, 3, 3],
                            'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 2, 'AUB_pbuffer': 2,
                                                       'BL0_pbuffer': 2, 'BL1_pbuffer': 1, 'BUB_pbuffer': 1,
                                                       'CL0_pbuffer': 1, 'CUB_pbuffer': 2, 'UBG_pbuffer': 2},
                                                       'n_bef_batch_flag': 0, 'n_bef_group_flag': 0,
                                                       'tbe_compile_para': 0}}

    input_list = [
        {'ori_shape': (2, 2, 2, 16, 16), 'shape': (2, 2, 2, 16, 16),
         'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'},
        {'ori_shape': (1, 1, 1, 1, 16), 'shape': (1, 1, 1, 1, 16),
         'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
        {'ori_shape': (1, 2, 2, 2, 16), 'shape': (1, 2, 2, 2, 16),
         'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
        (1, 2, 2, 2, 16), (1, 2, 2, 2, 1), [0, 0, 0, 0, 0, 0], (1, 1, 1, 1, 1),
        1, "NDHWC", "conv3d_dx_fault"
    ]

    TILING_INSTANCE.instance_refresh("tuning_tiling", tiling_params, tiling_dict)
    conv3d_backprop_input_d(*input_list)
    TILING_INSTANCE.instance_refresh(tiling_type, tiling_params, {})

ut_case.add_cust_test_func(test_func=test_conv3d_dx_mock_default_tiling)

# dsl d_dim wrong
def test_conv3d_dx_dsl_d_wrong(test_args):
    from tbe.dsl.compute import conv3d_backprop_input_compute as dx
    shape_filter = (2, 2, 2, 16, 16)
    shape_out_backprop = (1, 3, 1, 1, 16)
    input_sizes = (1, 2, 2, 2, 16)
    strides = (1, 2, 2, 2, 1)
    pads = [0, 0, 0, 0, 0, 0]
    dilations = (1, 1, 1, 1, 1)
    try:
        dx._check_conv3dbp_input_params_in_dsl(
            shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
            'float16', 'float16', 'float16', {})
    except RuntimeError as e:
        print(e)
ut_case.add_cust_test_func(test_func=test_conv3d_dx_dsl_d_wrong)

# dsl h_dim wrong
def test_conv3d_dx_dsl_h_wrong(test_args):
    from tbe.dsl.compute import conv3d_backprop_input_compute as dx
    shape_filter = (2, 2, 2, 16, 16)
    shape_out_backprop = (1, 1, 3, 1, 16)
    input_sizes = (1, 2, 2, 2, 16)
    strides = (1, 2, 2, 2, 1)
    pads = [0, 0, 0, 0, 0, 0]
    dilations = (1, 1, 1, 1, 1)
    try:
        dx._check_conv3dbp_input_params_in_dsl(
            shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
            'float16', 'float16', 'float16', {})
    except RuntimeError as e:
        print(e)
ut_case.add_cust_test_func(test_func=test_conv3d_dx_dsl_h_wrong)

# dsl w_dim wrong
def test_conv3d_dx_dsl_w_wrong(test_args):
    from tbe.dsl.compute import conv3d_backprop_input_compute as dx
    shape_filter = (2, 2, 2, 16, 16)
    shape_out_backprop = (1, 1, 1, 3, 16)
    input_sizes = (1, 2, 2, 2, 16)
    strides = (1, 2, 2, 2, 1)
    pads = [0, 0, 0, 0, 0, 0]
    dilations = (1, 1, 1, 1, 1)
    try:
        dx._check_conv3dbp_input_params_in_dsl(
            shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
            'float16', 'float16', 'float16', {})
    except RuntimeError as e:
        print(e)
ut_case.add_cust_test_func(test_func=test_conv3d_dx_dsl_w_wrong)

# dsl exceed L1
def test_conv3d_dx_dsl_exceed_l1(test_args):
    from tbe.dsl.compute import conv3d_backprop_input_compute as dx
    shape_filter = (7, 2, 2, 3, 19)
    shape_out_backprop = (4, 9, 2, 8, 19)
    input_sizes = (4, 31, 112, 118, 57)
    strides = (1, 3, 1, 1, 1)
    pads = [0, 0, 0, 0, 0, 0]
    dilations = (1, 1, 110, 110, 1)
    try:
        dx._check_conv3dbp_input_params_in_dsl(
            shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
            'float16', 'float16', 'float16', {})
    except RuntimeError as e:
        print(e)
ut_case.add_cust_test_func(test_func=test_conv3d_dx_dsl_exceed_l1)

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case1, "success", "case1", True))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 _gen_data_case(case2, "success", "case2", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case3, RuntimeError, "case3", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case4, RuntimeError, "case4", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case5, RuntimeError, "case5", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case6, RuntimeError, "case6", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case7, RuntimeError, "case7", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case8, RuntimeError, "case8", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case9, RuntimeError, "case9", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case10, RuntimeError, "case10", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case11, RuntimeError, "case11", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case12, RuntimeError, "case12", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case13, RuntimeError, "case13", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case14, RuntimeError, "case14", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case15, RuntimeError, "case15", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case16, RuntimeError, "case16", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case17, RuntimeError, "case17", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case18, RuntimeError, "case18", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case19, RuntimeError, "case19", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case20, RuntimeError, "case20", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case21, RuntimeError, "case21", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case22, RuntimeError, "case22", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case23, RuntimeError, "case23", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case24, RuntimeError, "case24", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case25, RuntimeError, "case25", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case26, RuntimeError, "case26", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case27, RuntimeError, "case27", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case28, RuntimeError, "case28", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(case_wo1, "success", "case_wo1", True))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
