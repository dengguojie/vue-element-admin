#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropFilterD", "impl.conv3d_backprop_filter_d",
               "conv3d_backprop_filter_d")


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _test_conv3d_dw_dsl_shape_error(test_arg):
    from impl.util.platform_adapter import tvm
    from tbe.dsl.compute.conv3d_backprop_filter_compute import Conv3dBackpropFilter
    # X have wrong shape
    try:
        input_x = tvm.placeholder([8, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [3, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = (1, 1, 1, 1, 1)
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_1()
    except Exception as e:
        print(e)
    # X have Wrong Value
    try:
        input_x = tvm.placeholder([8, -1, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [3, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = (1, 1, 1, 1, 1)
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_1()
    except Exception as e:
        print(e)
    
    # X C0 is not valid
    try:
        input_x = tvm.placeholder([8, 16, 2, 24, 24, 1], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [3, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = (1, 1, 1, 1, 1)
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_2()
    except Exception as e:
        print(e)
    
    # dedy C0 is not valid
    try:
        input_x = tvm.placeholder([8, 16, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 1], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [3, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = (1, 1, 1, 1, 1)
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_2()
    except Exception as e:
        print(e)
    
    # batch size is not Equal
    try:
        input_x = tvm.placeholder([4, 16, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [3, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = (1, 1, 1, 1, 1)
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_2()
    except Exception as e:
        print(e)


ut_case.add_cust_test_func(test_func=_test_conv3d_dw_dsl_shape_error)

# Test attribute Error
def _test_conv3d_dw_dsl_attr_error(test_arg):
    from impl.util.platform_adapter import tvm
    from tbe.dsl.compute.conv3d_backprop_filter_compute import Conv3dBackpropFilter
    # stride have wrong shape
    try:
        input_x = tvm.placeholder([8, 16, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [1, 3, 1, 3, 1]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = (1, 1, 1, 1, 1)
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_1()
    except Exception as e:
        print(e)
    # stride have wrong value
    try:
        input_x = tvm.placeholder([8, 16, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [65, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = (1, 1, 1, 1, 1)
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_1()
    except Exception as e:
        print(e)
    
    # Dilations have wrong value
    try:
        input_x = tvm.placeholder([8, 16, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [3, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = [2, 1, 1, 1, 2]
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_2()
    except Exception as e:
        print(e)
    
    # Dilations larger than 255
    try:
        input_x = tvm.placeholder([8, 16, 2, 24, 24, 16], name="fmap", dtype="float16")
        out_backprop = tvm.placeholder([8, 6, 4, 24, 8, 16], name="dedy", dtype="float16")
        filter_sizes = [56, 5, 5, 3, 3]
        strides =  [3, 1, 3]
        padding =  [2, 2, 3, 3, 1, 1]
        group_dict =  {'real_g': 1, 'mag_factor': 4, 'cin1_g': 2, 'cout_g': 64, 'cin_ori': 20, 'cout_ori': 56}
        dilations = [1, 256, 1, 1, 1]
        res_dtype =  "float32"
        deconv_dw_object = Conv3dBackpropFilter(input_x,
                                                out_backprop,
                                                filter_sizes,
                                                strides=strides,
                                                padding=padding,
                                                group_dict=group_dict,
                                                dilations=dilations,
                                                res_dtype=res_dtype,
                                                kernel_name="test_shape_error")
        deconv_dw_object._deconv_dw_input_check_2()
    except Exception as e:
        print(e)


ut_case.add_cust_test_func(test_func=_test_conv3d_dw_dsl_attr_error)

def test_op_check_supported(test_arg):
    from impl.conv3d_backprop_filter_d import check_supported
    out_backprop = {'ori_shape': (1, 5, 19, 75, 64), 'shape': (1, 5, 19, 75, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    (x_dict, out_backprop, y_input, filter_size, strides, pads, dilations,
        groups, data_format) = _run_api_end_with_d(out_backprop=out_backprop)
    check_supported(x_dict, out_backprop, y_input, filter_size, strides, pads, dilations, groups, data_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

def _test_op_get_op_support_info(test_arg):
    from impl.conv3d_backprop_filter_d import get_op_support_info

    [x_dict, out_backprop, y_input, filter_size, strides,
        pads, dilations, groups, data_format] = _run_api_end_with_d()

    get_op_support_info(
        x_dict, out_backprop, y_input, filter_size, strides,
        pads, dilations, groups, data_format)

    # test wrong_out_backprop format for get_info_op
    try:
        wrong_out_backprop = out_backprop.copy()
        wrong_out_backprop["ori_format"] = 'NHWC'
        get_op_support_info(
            x_dict, wrong_out_backprop, y_input, filter_size, strides,
            pads, dilations, groups, data_format)
    except Exception as e:
        print(e)

    # test wrong_y_dict format for get_info_op
    try:
        wrong_y_dict = y_input.copy()
        wrong_y_dict["ori_format"] = 'NHWC'
        get_op_support_info(
            x_dict, out_backprop, wrong_y_dict, filter_size, strides,
            pads, dilations, groups, data_format)
    except Exception as e:
        print(e)

    # test wrong_fmap format for get_info_op
    try:
        wrong_x_dict = x_dict.copy()
        wrong_x_dict["ori_format"] = 'NHWC'
        get_op_support_info(
            wrong_x_dict, out_backprop, y_input, filter_size, strides,
            pads, dilations, groups, data_format)
    except Exception as e:
        print(e)


ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)

# Test Schedule tiling constraints
def test_conv3d_dw_mock_tiling(test_args):
    from impl.conv3d_backprop_filter_d import conv3d_backprop_filter_d 
    from tbe.common.tiling.tiling_helper import TILING_INSTANCE
    tiling_type = "auto_tiling"
    tiling_params = {'a_shape': [8, 6, 4, 24, 8, 16], 'b_shape': [8, 16, 2, 24, 24, 16],
                     'c_shape': [64, 5, 3, 3, 32], 'a_dtype': 'float16', 'b_dtype': 'float16', 'c_dtype': 'float32',
                     'mad_dtype': 'float32', 'pad': [2, 2, 3, 3, 1, 1], 'stride': [3, 1, 3],
                     'strideh_expand': 1, 'stridew_expand': 1, 'dilation': [1, 3, 2], 'group': 1, 'fused_coefficient': [0, 0, 0],
                     'bias_flag': False, 'op_type': 'conv3d_backprop_filter',
                     'kernel_name': 'Conv3DBackpropFilterD_static_shape_schedule_case_ascend910a',
                     'model_type': 'xgboost', 'dynamic_shape_flag': False, 'fused_channel_wise': [0, 0, 0],
                     'fusion_type': 0, 'l1_fusion_type': -1, 'l2_fusion_type': -1, 'fm_l1_valid_size': 0, 'fm_l1_valid_size_level': 0}

    input_list = [
        {'ori_shape': (8, 16, 24, 24, 20), 'shape': (8, 16, 24, 24, 20),
         'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
        {'ori_shape': (8, 6, 24, 8, 56), 'shape': (8, 6, 24, 8, 56),
         'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
        {'ori_shape': (5, 3, 3, 5, 56), 'shape': (5, 3, 3, 5, 56),
         'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'},
        (5, 3, 3, 5, 56), (1, 3, 1, 3, 1), (2, 2, 3, 3, 1, 1), (1, 1, 3, 2, 1),
        4, "NDHWC", "conv3d_dw_constraint"
    ]
    tiling_dict_list = [
        # AL0 Pbuffer constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 0, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # AL1 Pbuffer constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 0, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # BL0_pbuffer constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 0,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # BL1_pbuffer constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 0, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # CL0_pbuffer constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 0, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # CUB_pbuffer constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 0, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # al1 is not al0 's integral multiple constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 2, 16, 16, 1, 1], 'AL1_shape': [1, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [2, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # al1 < 1 constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 0, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # (bl1_shape[0] // _CUBE_DIM) % bl0_matrix[0] constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 2, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [2, 9, 16, 16, 1, 1], 'BL1_shape': [16, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # bl1_shape[1] < 1 constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 0, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # al0_matrix[0] != cl0_matrix[1] constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 5, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 5, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # bl0_matrix[1] != cl0_matrix[0] constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 1, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [8, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [8, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # al0_matrix and bl0_matrix constraint
        {'conv3d_dw_constraint': 
            {'AL0_matrix': [4, 2, 16, 16, 1, 1], 'AL1_shape': [48, 1, 1, 1],
            'AUB_channel_wise_flag': None,
            'AUB_shape': [2, 0, 0, 1], 'A_overhead_opt_flag': 0,
            'BL0_matrix': [1, 9, 16, 16, 1, 1], 'BL1_shape': [96, 1, 1, 1],
            'BUB_channel_wise_flag': None, 'BUB_shape': [1, 0, 0, 1],
            'B_overhead_opt_flag': 0, 'CL0_matrix': [9, 4, 16, 16, 1, 1],
            'CUB_channel_wise_flag': False, 'CUB_matrix': [9, 4, 16, 16, 1, 1],
            'batch_bef_group_flag': 0, 'block_dim': [24, 1, 1, 1],
            'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}}
    ]
    for tiling_dict in tiling_dict_list:
        try :
            TILING_INSTANCE.instance_refresh("tuning_tiling", tiling_params, tiling_dict)
            conv3d_backprop_filter_d(*input_list)
        except RuntimeError as e:
            print(e)
        finally:
            TILING_INSTANCE.instance_refresh(tiling_type, tiling_params, {})


ut_case.add_cust_test_func(test_func=test_conv3d_dw_mock_tiling)

def _run_api_end_with_d(
    x_dict={'ori_shape': (1, 16, 120, 176, 32), 'shape': (1, 16, 120, 176, 32),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    out_backprop={'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
                  'ori_format': 'NDHWC', 'format': 'NDHWC',
                  'dtype': 'float16'},
    y_input={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
             'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'},
    filter_size=(2, 2, 2, 32, 64),
    strides=(1, 2, 2, 2, 1),
    pads=(0, 0, 0, 0, 0, 0),
    dilations=(1, 1, 1, 1, 1),
    groups=1,
    data_format="NDHWC"):
    return [x_dict, out_backprop, y_input, filter_size, strides,
            pads, dilations, groups, data_format]


# Define Cases instances
# test_conv3dbp_filter_succ
success_case1 = _run_api_end_with_d()

# test_conv3dbp_filter_stride_one
strides = (1, 1, 1)
out_backprop = {'ori_shape': (1, 15, 119, 175, 64),
                'shape': (1, 15, 119, 175, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
stride_one_success_case = _run_api_end_with_d(out_backprop=out_backprop, strides=strides)

# test_w_res_not_match_constraint
out_backprop = {'ori_shape': (1, 5, 19, 75, 64), 'shape': (1, 5, 19, 75, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
w_res_not_match_case = _run_api_end_with_d(out_backprop=out_backprop)

# test_conv3dbp_filter_invalid_pads_length
pads = [0, 0, 0, 0]
invalid_pads_length_case = _run_api_end_with_d(pads=pads)

# test_conv3dbp_filter_invalid_dilations_length
dilations = [1, 0, 1, 0]
invalid_dilations_length_case = _run_api_end_with_d(dilations=dilations)

# test_conv3dbp_filter_invalid_shape
y_input = {'ori_shape': (2, 2, 16, 64), 'shape': (2, 2, 16, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
invalid_filter_shape_case = _run_api_end_with_d(y_input=y_input)

# test_conv3dbp_fmap_invalid_shape
x_dict = {'ori_shape': (16, 120, 176, 32), 'shape': (16, 120, 176, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
invalid_fmap_shape_case = _run_api_end_with_d(x_dict=x_dict)

# test_conv3dp_filter_fmap_invalid_input_type
x_dict = {'ori_shape': (1, 16, 120, 176, 32), 'shape': (1, 16, 120, 176, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float32'}
invalid_fmap_dtype_case = _run_api_end_with_d(x_dict=x_dict)

# test_conv3dp_filter_dedy_invalid_input_type
out_backprop = {'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float32'}
invalid_dedy_dtype_case = _run_api_end_with_d(out_backprop=out_backprop)

# test_wrong_x_format
wrong_x_dict = {'ori_shape': (1, 16, 120, 176, 32), 'shape': (1, 16, 120, 176, 32),
                'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float32'}
invalid_fmap_format_case = _run_api_end_with_d(x_dict=wrong_x_dict)

# test_conv3dbp_filter_invalid_shape_length
out_backprop = {'ori_shape': (1, 8, 60, 88), 'shape': (1, 8, 60, 88),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
invalid_dedy_shape_case = _run_api_end_with_d(out_backprop=out_backprop)

# test_conv3dbp_filter_invalid_stides_length
strides = (1, 1, 1, 1)
invalid_stride_length_case = _run_api_end_with_d(strides=strides)

# test_conv3dbp_filter_invalid_filter_size_length
filter_size = (2, 2, 2, 64)
invalid_filter_size_case = _run_api_end_with_d(filter_size=filter_size)

# test_conv3dbp_filter_invalid_pads
pads = {"2": 2}
invalid_pad_type_case = _run_api_end_with_d(pads=pads)

# test_conv3dbp_filter_invalid_dilations_value
dilations = [2, 1, 1, 1, 2]
invalid_dilation_n_case = _run_api_end_with_d(dilations=dilations)

# test_conv3dbp_filter_dedy_format_failed
out_backprop = {'ori_shape': (1, 5, 19, 75, 64), 'shape': (1, 5, 19, 75, 64),
                'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float32'}
invalid_dedy_format_case = _run_api_end_with_d(out_backprop=out_backprop)

# test_conv3dbp_filter_wrong_filter_batch
filter_size = (2, 2, 2, 32, 32)
filter_batch_not_match_case = _run_api_end_with_d(filter_size=filter_size)

# test_conv3dbp_filter_wrong_filter_channel
filter_size = (2, 2, 2, 22, 64)
filter_channel_not_match_case = _run_api_end_with_d(filter_size=filter_size)

# test_conv3dbp_filter_wrong_fmap_batch
x_dict = {'ori_shape': (2, 16, 120, 176, 32), 'shape': (2, 16, 120, 176, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
x_batch_not_match_case = _run_api_end_with_d(x_dict=x_dict)

# Success Case
out_backprop = {'ori_shape': (1, 9, 61, 89, 64), 'shape': (1, 9, 61, 89, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC',
                'dtype': 'float16'}
pads = (1, 1, 1, 1, 1, 1)
success_case2 = _run_api_end_with_d(out_backprop=out_backprop, pads=pads)

# test_conv3dbp_filter Not flag all one
x_dict = {'ori_shape': (1, 32, 240, 352, 16), 'shape': (1, 32, 240, 352, 16),
          'ori_format': 'NDHWC', 'format': 'NDHWC',
          'dtype': 'float16'}
out_backprop = {'ori_shape': (1, 32, 240, 352, 16), 'shape': (1, 32, 240, 352, 16),
                'ori_format': 'NDHWC', 'format': 'NDHWC',
                'dtype': 'float16'}
y_input = {'ori_shape': (3, 3, 3, 16, 16), 'shape': (3, 3, 3, 16, 16),
           'ori_format': 'DHWCN', 'format': 'DHWCN',
           'dtype': 'float32'}
filter_size = (3, 3, 3, 16, 16)
pads = (1, 1, 1, 1, 1, 1)
strides = (1,1,1,1,1)
not_flag_all_one_success_case = _run_api_end_with_d(x_dict=x_dict, out_backprop=out_backprop,
                             y_input=y_input, filter_size=filter_size, pads=pads, strides=strides)

# test check range fail_stride
strides = (1, 1, 65, 1,1)
invalid_stride_range_case = _run_api_end_with_d(strides=strides)

# test padding larger than filter
out_backprop = {'ori_shape': (1, 12, 60, 88, 64), 'shape': (1, 12, 60, 88, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC',
                'dtype': 'float16'}
err_pads = (4, 4, 0, 0, 0, 0)
pad_d_larger_than_filter_case = _run_api_end_with_d(out_backprop=out_backprop, pads=err_pads)

out_backprop = {'ori_shape': (1, 8, 64, 88, 64), 'shape': (1, 8, 64, 88, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC',
                'dtype': 'float16'}
err_pads = (0, 0, 4, 4, 0, 0)
pad_h_larger_than_filter_case = _run_api_end_with_d(out_backprop=out_backprop, pads=err_pads)

out_backprop = {'ori_shape': (1, 8, 60, 92, 64), 'shape': (1, 8, 60, 92, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC',
                'dtype': 'float16'}
err_pads = (0, 0, 0, 0, 4, 4)
pad_w_larger_than_filter_case = _run_api_end_with_d(out_backprop=out_backprop, pads=err_pads)

# test_wrong_dedy_h dim value check
out_backprop = {'ori_shape': (1, 8, 233, 88, 64), 'shape': (1, 8, 233, 88, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC',
                'dtype': 'float16'}
dedy_h_not_match_case = _run_api_end_with_d(out_backprop=out_backprop)

# test_not_match_dedy_c
wrong_out_backprop = {'ori_shape': (1, 8, 60, 88, 1), 'shape': (1, 8, 60, 88, 1),
                      'ori_format': 'NDHWC', 'format': 'NDHWC',
                      'dtype': 'float16'}
test_not_match_dedy_c = _run_api_end_with_d(out_backprop=wrong_out_backprop)

# test filter_h_dilation > fmap_h_padding
y_input = {'ori_shape': (2, 255, 2, 32, 64), 'shape': (2, 255, 2, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
filter_h_larger_than_fmap_h_case = _run_api_end_with_d(y_input=y_input)

# test filter_w_dilation > fmap_w_padding
y_input = {'ori_shape': (2, 2, 255, 32, 64), 'shape': (2, 2, 255, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
test_filter_fmap_w_relation_case = _run_api_end_with_d(y_input=y_input)

# test fmap_channel != filter_channel * groups: 
invalid_groups_case = _run_api_end_with_d(groups=2)

# Test Load3D_padding_case
x_dict = {'ori_shape': (29, 4, 682, 1, 37), 'shape': (29, 4, 682, 1, 37),
           'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
out_backprop = {'ori_shape': (29, 1, 27, 1, 28), 'shape': (29, 1, 27, 1, 28),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
y_input = {'ori_shape': (1, 4, 4, 37, 28), 'shape': (1, 4, 4, 37, 28),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
filter_size = (1, 4, 4, 37, 28)
strides = (1, 44, 26, 47, 1)
padding = [0, 0, 5, 5, 1, 2]
dilations = (1, 1, 5, 1, 1)
load3d_padding_case = _run_api_end_with_d(x_dict=x_dict, out_backprop=out_backprop,
                                          y_input=y_input, filter_size=filter_size, 
                                          strides=strides, pads=padding, dilations=dilations)

# Test flag all one case
x_dict = {'ori_shape': (4, 16, 1, 1, 256), 'shape': (4, 16, 1, 1, 256),
           'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
out_backprop = {'ori_shape': (4, 16, 1, 1, 80), 'shape': (4, 16, 1, 1, 80),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
y_input = {'ori_shape': (1, 1, 1, 256, 80), 'shape': (1, 1, 1, 256, 80),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
filter_size = (1, 1, 1, 256, 80)
strides = (1, 1, 1, 1, 1)
load2d_case = _run_api_end_with_d(x_dict=x_dict, out_backprop=out_backprop,
                                  y_input=y_input, filter_size=filter_size, 
                                  strides=strides)

# Test grads L1 tiling part larger than fmal l1 tiling part
x_dict = {'ori_shape': (8, 16, 24, 24, 20), 'shape': (8, 16, 24, 24, 20),
           'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
out_backprop = {'ori_shape': (8, 6, 24, 8, 56), 'shape': (8, 6, 24, 8, 56),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
y_input = {'ori_shape': (5, 3, 3, 5, 56), 'shape': (5, 3, 3, 5, 56),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
filter_size = (5, 3, 3, 5, 56)
strides = (1, 3, 1, 3, 1)
pads = (2, 2, 3, 3, 1, 1)
dilations = (1, 1, 3, 2, 1)
schedule_case = _run_api_end_with_d(x_dict=x_dict, out_backprop=out_backprop,
                                    y_input=y_input, filter_size=filter_size, 
                                    strides=strides, dilations=dilations, pads=pads, groups=4)

    
# Add test Cases
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(success_case1, "success", "success_case1", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(stride_one_success_case, "success", "stride_one_success_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(w_res_not_match_case, RuntimeError, "w_res_not_match_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_pads_length_case, RuntimeError, "invalid_pads_length_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_dilations_length_case, RuntimeError, "invalid_dilations_length_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_filter_shape_case, RuntimeError, "invalid_filter_shape_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_fmap_shape_case, RuntimeError, "invalid_fmap_shape_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_fmap_dtype_case, RuntimeError, "invalid_fmap_dtype_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_dedy_dtype_case, RuntimeError, "invalid_dedy_dtype_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_fmap_format_case, RuntimeError, "invalid_fmap_format_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_dedy_shape_case, RuntimeError, "invalid_dedy_shape_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_stride_length_case, RuntimeError, "invalid_stride_length_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_filter_size_case, RuntimeError, "invalid_filter_size_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_pad_type_case, RuntimeError, "invalid_pad_type_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_dilation_n_case, RuntimeError, "invalid_dilation_n_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_dedy_format_case, RuntimeError, "invalid_dedy_format_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(filter_batch_not_match_case, RuntimeError, "filter_batch_not_match_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(filter_channel_not_match_case, RuntimeError, "filter_channel_not_match_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(x_batch_not_match_case, RuntimeError, "x_batch_not_match_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(success_case2, "success", "success_case2", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(not_flag_all_one_success_case, "success", "not_flag_all_one_success_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_stride_range_case, RuntimeError, "invalid_stride_range_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(pad_d_larger_than_filter_case, RuntimeError, "pad_d_larger_than_filter_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(pad_h_larger_than_filter_case, RuntimeError, "pad_h_larger_than_filter_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(pad_w_larger_than_filter_case, RuntimeError, "pad_w_larger_than_filter_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dedy_h_not_match_case, RuntimeError, "dedy_h_not_match_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(test_not_match_dedy_c, RuntimeError, "test_not_match_dedy_c", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(filter_h_larger_than_fmap_h_case, RuntimeError, "filter_h_larger_than_fmap_h_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(test_filter_fmap_w_relation_case, RuntimeError, "test_filter_fmap_w_relation_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(invalid_groups_case, RuntimeError, "invalid_groups_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(load3d_padding_case, "success", "load3d_padding_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(load2d_case, "success", "load2d_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(schedule_case, "success", "schedule_case", True))

if __name__ == '__main__':
    ut_case.run('Ascend910A')
    exit(0)