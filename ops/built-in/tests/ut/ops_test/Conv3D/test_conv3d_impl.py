#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3D
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3D", "impl.conv3d", "conv3d")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api_end_with_d(
    fmap={'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 60, 88, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    weight={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'},
    bias=None, offset_w=None,
    output={'ori_shape': (1, 4, 30, 44, 64), 'shape': (1, 4, 30, 44, 64),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    strides=(1, 2, 2, 2, 1),
    pads=[0, 0, 0, 0, 0, 0],
    dilations=(1, 1, 1, 1, 1),
    groups=1, data_format="NDHWC", offset_x=0):
    return [fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x]


def test_op_check_supported(test_arg):
    from impl.conv3d import check_supported
    fmap = {'ori_shape': (2, 32, 15, 4098, 18), 'shape': (2, 32, 15, 4098, 18),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    (fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, _) = _run_api_end_with_d(fmap = fmap)
    check_supported(fmap, weight, bias, offset_w, output, strides, pads, dilations, groups, data_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

def _test_op_get_op_support_info(test_arg):
    from impl.conv3d import get_op_support_info
    (fmap, weight, bias, offset_w, output, strides,
        pads, dilations, groups, data_format, offset_x) = _run_api_end_with_d()
    get_op_support_info(
       fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x)
    # Test Bias Cut in NDC1HWCO
    fmap_ndc1hwc0 = {'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 2, 60, 88, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'}
    bias = {'ori_shape': (64,), 'shape': (64,),
            'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    get_op_support_info(
       fmap_ndc1hwc0, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x)
    # Test None filter shape runtime Errors
    wrong_filter = weight.copy()
    wrong_filter['ori_format'] = 'NDDDD'
    wrong_filter['format'] = 'NDDDD'
    try:
        get_op_support_info(
        fmap, wrong_filter, None, offset_w, output, strides,
                pads, dilations, groups, data_format, offset_x)
    except Exception as e:
        print(e)
    # Test strides_formated runtime Errors
    wrong_fmap = fmap.copy()
    wrong_fmap['ori_format'] = 'NDDDD'
    wrong_fmap['format'] = 'NDDDD'
    try:
        get_op_support_info(
        wrong_fmap, weight, None, offset_w, output, strides,
                pads, dilations, groups, data_format, offset_x)
    except Exception as e:
        print(e)
    
ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)

# test _check_d_dimension
def _test_check_d_dimension(test_arg):
    from tbe.dsl.compute.conv3d_compute import _check_d_dimension
    try:
        # Check Filter range
        _check_d_dimension(8, 512, [0, 0], 2, 1)
    except Exception as e:
        print(e)

    try:
        # Check (fmap_d + pad_d[0] + pad_d[1]) < filter_dilated_d
        _check_d_dimension(2, 3, [0, 0], 2, 1)
    except Exception as e:
        print(e)

    try:
        # Check pad_d[0] > _PAD_MAX or pad_d[1] > _PAD_MAX
        _check_d_dimension(8, 2, [256, 256], 2, 1)
    except Exception as e:
        print(e)

    try:
        # pad_d[0] >= filter_dilated_d or pad_d[1] >= filter_dilated_d
        _check_d_dimension(8, 2, [4, 4], 2, 1)
    except Exception as e:
        print(e)

    try:
        # stride_d < _STRIDE_MIN or stride_d > _STRIDE_MAX
        _check_d_dimension(8, 2, [0, 0], 65, 1)
    except Exception as e:
        print(e)


ut_case.add_cust_test_func(test_func=_test_check_d_dimension)

# test _check_h_dimension
def _test_check_h_dimension(test_arg):
    from tbe.dsl.compute.conv3d_compute import _check_h_dimension
    try:
        # Check Filter range
        _check_h_dimension(8, 512, [0, 0], 2, 1)
    except Exception as e:
        print(e)

    try:
        # Check (fmap_h + pad_h[0] + pad_h[1]) < filter_dilated_h
        _check_h_dimension(2, 3, [0, 0], 2, 1)
    except Exception as e:
        print(e)

    try:
        # Check pad_h[0] > _PAD_MAX or pad_h[1] > _PAD_MAX
        _check_h_dimension(8, 2, [256, 256], 2, 1)
    except Exception as e:
        print(e)

    try:
        # pad_h[0] >= filter_dilated_h or pad_h[1] >= filter_dilated_h
        _check_h_dimension(8, 2, [4, 4], 2, 1)
    except Exception as e:
        print(e)

    try:
        # stride_h < _STRIDE_MIN or stride_h > _STRIDE_MAX
        _check_h_dimension(8, 2, [0, 0], 65, 1)
    except Exception as e:
        print(e)


ut_case.add_cust_test_func(test_func=_test_check_h_dimension)

# test _check_w_dimension
def _test_check_w_dimension(test_arg):
    from tbe.dsl.compute.conv3d_compute import _check_w_dimension
    try:
        # Check Filter range
        _check_w_dimension(8, 512, [0, 0], 2, 1)
    except Exception as e:
        print(e)

    try:
        # Check (fmap_w + pad_w[0] + pad_w[1]) < filter_dilated_w
        _check_w_dimension(2, 3, [0, 0], 2, 1)
    except Exception as e:
        print(e)

    try:
        # Check pad_w[0] > _PAD_MAX or pad_w[1] > _PAD_MAX
        _check_w_dimension(8, 2, [256, 256], 2, 1)
    except Exception as e:
        print(e)

    try:
        # pad_w[0] >= filter_dilated_w or pad_w[1] >= filter_dilated_w
        _check_w_dimension(8, 2, [4, 4], 2, 1)
    except Exception as e:
        print(e)

    try:
        # stride_w < _STRIDE_MIN or stride_w > _STRIDE_MAX
        _check_w_dimension(8, 2, [0, 0], 65, 1)
    except Exception as e:
        print(e)


ut_case.add_cust_test_func(test_func=_test_check_w_dimension)

def _test_check_conv3d_dtype(test_arg):
    from tbe.dsl.compute.conv3d_compute import _check_conv3d_dtype
    # Fmap Error
    try:
        _check_conv3d_dtype('int8', 'float16', 'float32')
    except Exception as e:
        print(e)
    # Filter Error
    try:
        _check_conv3d_dtype('float16', 'int8', 'float32')
    except Exception as e:
        print(e)
    # output Error
    try:
        _check_conv3d_dtype('float16', 'float16', 'int8')
    except Exception as e:
        print(e)
ut_case.add_cust_test_func(test_func=_test_check_conv3d_dtype)

# Test Schedule check Tiling
def test_conv3d_mock_tiling(test_args):
    from impl.conv3d import conv3d
    from tbe.common.tiling.tiling_helper import TILING_INSTANCE
    tiling_type = "auto_tiling"
    tiling_params = {'op_type': 'convolution_3d',
        'a_shape': [1, 8, 2, 60, 88, 16], 'b_shape': [64, 2, 2, 2, 2, 16],
        'a_dtype': 'float16', 'b_dtype': 'float16', 'c_dtype': 'float16',
        'mad_dtype': 'float32', 'pad': [0, 0, 0, 0, 0, 0], 'stride': [2, 2, 2],
        'dilation': [1, 1, 1], 'bias_flag': False, 'fused_coefficient': [0, 0, 0],
        'group': 1, 'kernel_name': 'Conv3D_static_shape_case1_ascend910a',
        'model_type': 'xgboost', 'c_shape': [0, 0, 0, 0, 0], 'strideh_expand': 1, 'stridew_expand': 1,
        'dynamic_shape_flag': False, 'fused_channel_wise': [0, 0, 0],
        'fusion_type': 0, 'l1_fusion_type': -1, 'l2_fusion_type': -1,
        'fm_l1_valid_size': 0, 'fm_l1_valid_size_level': 0}

    input_list = [
        {'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 60, 88, 32),
         'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
        {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
         'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}, None, None,
        {'ori_shape': (1, 4, 30, 44, 64), 'shape': (1, 4, 30, 44, 64),
         'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
        (1, 2, 2, 2, 1), [0, 0, 0, 0, 0, 0], (1, 1, 1, 1, 1),
        1, "NDHWC", 0, "conv3d_tiling_test"
    ]
    
    tiling_dict_list = [
        # AL0 not equal CL0
        {'conv3d_tiling_test': 
            {'AL0_matrix': [56, 1, 16, 16, 1, 1], 'AL1_shape': [128, 1, 1, 2],
            'AUB_channel_wise_flag': None, 'AUB_shape': None, 'A_overhead_opt_flag': 0,
            'BL0_matrix': [], 'BL1_shape': None, 'BUB_channel_wise_flag': None,
            'BUB_shape': [1, 0, 0, 0], 'B_overhead_opt_flag': 0,
            'CL0_matrix': [2, 2, 16, 16, 1, 1], 'CUB_channel_wise_flag': False,
            'CUB_matrix': [2, 2, 16, 16, 1, 1], 'batch_bef_group_flag': 0,
            'block_dim': [1, 2, 4, 4], 'manual_pingpong_buffer': {'AL0_pbuffer': 2,
            'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2,
            'CUB_pbuffer': 2, 'UBG_pbuffer': 2},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # AL0 m_C0 not equal to 16
        {'conv3d_tiling_test': 
            {'AL0_matrix': [56, 1, 8, 16, 1, 1], 'AL1_shape': [128, 1, 1, 2],
            'AUB_channel_wise_flag': None, 'AUB_shape': None, 'A_overhead_opt_flag': 0,
            'BL0_matrix': [], 'BL1_shape': None, 'BUB_channel_wise_flag': None,
            'BUB_shape': [1, 0, 0, 0], 'B_overhead_opt_flag': 0,
            'CL0_matrix': [2, 56, 16, 16, 1, 1], 'CUB_channel_wise_flag': False,
            'CUB_matrix': [2, 56, 16, 16, 1, 1], 'batch_bef_group_flag': 0,
            'block_dim': [1, 2, 4, 4], 'manual_pingpong_buffer': {'AL0_pbuffer': 2,
            'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2,
            'CUB_pbuffer': 2, 'UBG_pbuffer': 2},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}},
        # AL0 k_C0 not equal to 16
        {'conv3d_tiling_test': 
            {'AL0_matrix': [56, 1, 16, 8, 1, 1], 'AL1_shape': [128, 1, 1, 2],
            'AUB_channel_wise_flag': None, 'AUB_shape': None, 'A_overhead_opt_flag': 0,
            'BL0_matrix': [], 'BL1_shape': None, 'BUB_channel_wise_flag': None,
            'BUB_shape': [1, 0, 0, 0], 'B_overhead_opt_flag': 0,
            'CL0_matrix': [2, 56, 16, 16, 1, 1], 'CUB_channel_wise_flag': False,
            'CUB_matrix': [2, 56, 16, 16, 1, 1], 'batch_bef_group_flag': 0,
            'block_dim': [1, 2, 4, 4], 'manual_pingpong_buffer': {'AL0_pbuffer': 2,
            'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1,
            'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2,
            'CUB_pbuffer': 2, 'UBG_pbuffer': 2},
            'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}}
    ]
    for tiling_dict in tiling_dict_list:
        try :
            TILING_INSTANCE.instance_refresh("tuning_tiling", tiling_params, tiling_dict)
            conv3d(*input_list)
        except RuntimeError as e:
            print(e)
        finally:
            TILING_INSTANCE.instance_refresh(tiling_type, tiling_params, {})

ut_case.add_cust_test_func(test_func=test_conv3d_mock_tiling)

# Ho!=1Wo=1 for ub
def test_conv3d_mul_fusion_load3d_special_case(test_args):
    from tvm.target import cce
    from impl.util.platform_adapter import tvm
    from impl.util.platform_adapter import tbe
    from impl.conv3d import conv3d_fusion_compute
    _, _, bias, offset_w, output, _,\
        _, _, _, data_format, offset_x = _run_api_end_with_d()
    strides = [1, 51, 26, 12, 1]
    pads = [0, 0, 0, 0, 0, 0]
    fmap = tvm.placeholder((52, 3, 1, 1855, 4, 16),
                             name="filter",
                             dtype="float16",
                             attrs={"ori_shape": (52, 3, 1855, 4, 12),
                                    "ori_format": "NDHWC",
                                    "data_type": "float16"})
    weight = tvm.placeholder((48, 4, 16, 16),
                            name="grads",
                            dtype="float16",
                            attrs={"ori_shape": (3, 4, 4, 12, 59),
                                   "ori_format": "DHWCN",
                                   "data_type": "float16"})
    mul_tensor = tvm.placeholder((52, 4, 72, 16),
                                 name="mul",
                                 dtype="float16",
                                 attrs={"ori_shape": (52, 4, 72, 16),
                                        "ori_format": "NDHWC",
                                        "data_type": "float16"})

    res = conv3d_fusion_compute(fmap,
                          weight,
                          bias,
                          offset_w,
                          output,
                          strides,
                          pads,
                          dilations=(1, 1, 1, 1, 1),
                          groups=1,
                          data_format="NDHWC",
                          offset_x=0,
                          kernel_name="conv3d")

    mul_res = tbe.vmul(res, mul_tensor)
    tensor_list = [fmap, weight, mul_tensor, mul_res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(mul_res)

    config = {
            "name": "test_conv3d_mul_fusion_load3d_special_case",
            "tensor_list": tensor_list,
        }
    tbe.build(sch, config)
ut_case.add_cust_test_func(test_func=test_conv3d_mul_fusion_load3d_special_case)

# test_conv3d_succ_d
success_case1 = _run_api_end_with_d()

# test_conv3d_stride_one
output = {'ori_shape': (1, 7, 59, 87, 64), 'shape': (1, 7, 59, 87, 64),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
strides_one_success_case = _run_api_end_with_d(output=output, strides=strides)

# test_bias_length_fail
bias = {'ori_shape': (64, 64,), 'shape': (64, 64,),
        'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
bias_length_fail_case = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (2, 32, 15, 4098, 18), 'shape': (2, 32, 15, 4098, 18),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
invalid_fmap_w_shape = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_output
output = {'dtype': 'float32'}
invalid_output_case = _run_api_end_with_d(output=output)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (1, 8, 60, 88), 'shape': (1, 8, 60, 88),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
invalid_fmap_shape_case = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_pad_length
pads = (0, -1, -1, -1, 0)
invalid_pad_length = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 2, 32), 'shape': (2, 2, 2, 32),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
invalid_weight_length_case = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight_w
weight = {'ori_shape': (2, 2, 354, 32, 64), 'shape': (2, 2, 354, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
invalid_weight_w_case = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_big_fmap
fmap = {'ori_shape': (200, 3000, 4000, 4000, 3000),
      'shape': (200, 3000, 4000, 4000, 3000),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
too_large_fmap_case = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_bias_dtype
bias = {'ori_shape': (1,), "dtype": "float32"}
invalid_bias_type_case = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_stride_d_shape
strides = (1, 0, 1, 1, 1)
invalid_stride_d_case = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_h_shape
strides = (1, 1, 0, 1, 1)
invalid_stride_h_case = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_w_shape
strides = (1, 1, 1, 0, 1)
invalid_stride_w_case = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_weight_d
weight = {'ori_shape': (257, 2, 2, 32, 64), 'shape': (257, 2, 2, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
invalid_weight_d_case = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight_h
weight = {'ori_shape': (2, 257, 2, 32, 64), 'shape': (2, 257, 2, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
invalid_weight_h_case = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (1, 8, 60, 4098, 32), 'shape': (1, 8, 60, 4098, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
invalid_fmap_w_case = _run_api_end_with_d(fmap=fmap)

fmap = {'ori_shape': (1, 8, 4098, 88, 32), 'shape': (1, 8, 4098, 88, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
invalid_fmap_h_case = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 257, 32, 64), 'shape': (2, 2, 257, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
invalid_weihgt_w_case = _run_api_end_with_d(weight=weight)

# Test Conv3D Bias Case
bias = {'ori_shape': (64,), 'shape': (64,),
            'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
bias_success_case = _run_api_end_with_d(bias=bias)

# test_conv3d_fmap_wrong_format
# WARNING: Did not trigger anything. This is consider to be redundant
wrong_fmap={'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 60, 88, 32),
            'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
fmap_format_wrong_case = _run_api_end_with_d(fmap=wrong_fmap)

# test_conv3d_invalid_fmap_format
fmap = {'ori_shape': (1, 32, 8, 60, 88), 'shape': (1, 32, 8, 60, 88),
      'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16'}
invalid_fmap_format_case = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_weight_format_case
weight = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
        'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16'}
invalid_weight_format_case1 = _run_api_end_with_d(weight=weight)

# test_conv3d_wight_wrong_format
wrong_weight = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
              'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
invalid_weight_format_case2 = _run_api_end_with_d(weight=wrong_weight)

# test_stride_length_constraint
wrong_strides = [1,1,1]
invalid_stride_length_case = _run_api_end_with_d(strides=wrong_strides)

# test_dilation_length_constraint
wrong_dilations = [1,1,1]
invalid_dilations_length_case = _run_api_end_with_d(dilations=wrong_dilations)

# test_groups_constraint
invalid_groups_case = _run_api_end_with_d(groups=2)

# test_conv3d_invalid_dilations
dilations = (1, 2, 1, 1, 1)
invalid_dilation_d_case = _run_api_end_with_d(dilations=dilations)

# test_conv3d_dilation_d_zero
dilations = (1, 0, 1, 1, 1)
invalid_dilation_d_zero_case = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_dilations_shape
dilations = (1, 1, 0, 1, 1)
invalid_dilation_h_case = _run_api_end_with_d(dilations=dilations)

# test_conv3d_dilation_w_zero
dilations = (1, 1, 1, 0, 1)
invalid_dilation_w_case = _run_api_end_with_d(dilations=dilations)

# test_conv3d_fmap_wrong_format
wrong_format = "NHWC"
invalid_format_case = _run_api_end_with_d(data_format=wrong_format)

# test_fmap_dim + pad < filter_dim
fmap = {'ori_shape': (1, 8, 8, 8, 32), 'shape': (1, 8, 8, 8, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (10, 10, 10, 32, 64), 'shape': (10, 10, 10, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
test_fmap_filter_relation_case1 = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides)

fmap = {'ori_shape': (1, 8, 8, 8, 32), 'shape': (1, 8, 8, 8, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (2, 10, 10, 32, 64), 'shape': (2, 10, 10, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
test_fmap_filter_relation_case2 = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides)

fmap = {'ori_shape': (1, 8, 8, 8, 32), 'shape': (1, 8, 8, 8, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (2, 2, 10, 32, 64), 'shape': (2, 2, 10, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
test_fmap_filter_relation_case3 = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides)

# Test Padding Error

# test_conv3d_invalid_pad_d_range
pads = (256, 256, 256, 256, 256, 256)
invalid_pads_d_range_case = _run_api_end_with_d(pads=pads)

# test_conv3d_pad_d_larger than filter
pads = (3, 3, 0, 0, 0, 0)
invalid_pad_d_case = _run_api_end_with_d(pads=pads)

# Invalid pad_h
pads = (0, 0, 256, 256, 256, 256)
invalid_pad_h_range_case = _run_api_end_with_d(pads=pads)

pads = (0, 0, 3, 3, 0, 0)
invalid_pad_h_case = _run_api_end_with_d(pads=pads)

# invalid pads_w_case
pads = (1, 1, 1, 1, 3, 1)
invalid_pads_w_case = _run_api_end_with_d(pads=pads)

pads = (0, 0, 0, 0, 256, 256)
invalid_pad_w_range_case = _run_api_end_with_d(pads=pads)

# Test Special Load3D Case
fmap = {'ori_shape': (43, 98, 346, 1, 37), 'shape': (43, 98, 346, 1, 37),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (2, 1, 1, 1, 185), 'shape': (2, 1, 1, 1, 185),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 51, 17, 43, 1)
pads = (0, 0, 0, 0, 0, 0)
dilations = (1, 1, 15, 6, 1)
load3D_padding_case = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides,
                                          pads=pads, dilations=dilations, groups=37)

# Test Case With Default Tiling
fmap = {'ori_shape': (2, 4, 24, 24, 4420), 'shape': (2, 4, 24, 24, 4420),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (1, 3, 2, 1, 4420), 'shape': (1, 3, 2, 1, 4420),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 5, 3, 3, 1)
dilations = (1, 1, 6, 2, 1)
default_tiling_case = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides,
                                          dilations=dilations, groups=4420)

# Test Case With Default Tiling
fmap = {'ori_shape': (64, 4, 24, 24, 4420), 'shape': (64, 4, 24, 24, 4420),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (1, 3, 2, 1, 4420), 'shape': (1, 3, 2, 1, 4420),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 5, 3, 3, 1)
dilations = (1, 1, 6, 2, 1)
large_batch_default_tiling_case = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides,
                                          dilations=dilations, groups=4420)

# Test Load2D Case
fmap = {'ori_shape': (64, 2, 1, 1, 256), 'shape': (64, 2, 1, 1, 256),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (1, 1, 1, 256, 64), 'shape': (1, 1, 1, 256, 64),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
load2d_case = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides)

# Test CycleBUffer Flag
fmap = {'ori_shape': (2, 128, 128, 1128, 4), 'shape': (2, 128, 128, 1128, 4),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
weight = {'ori_shape': (3, 3, 3, 4, 32), 'shape': (3, 3, 3, 4, 32),
          'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
pads = [1, 1, 1, 1, 1, 1]
cycle_buffer_case = _run_api_end_with_d(fmap=fmap, weight=weight, strides=strides, pads=pads)

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(success_case1, "success", "success_case1", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(strides_one_success_case, "success", "strides_one_success_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(bias_length_fail_case, RuntimeError, "bias_length_fail_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_fmap_w_shape, RuntimeError, "invalid_fmap_w_shape", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_output_case, RuntimeError, "invalid_output_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_fmap_shape_case, RuntimeError, "invalid_fmap_shape_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_pad_length, RuntimeError, "invalid_pad_length", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_weight_length_case, RuntimeError, "invalid_weight_length_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_weight_w_case, RuntimeError, "invalid_weight_w_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(too_large_fmap_case, RuntimeError, "too_large_fmap_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_bias_type_case, RuntimeError, "invalid_bias_type_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_pads_w_case, RuntimeError, "invalid_pads_w_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_fmap_format_case, RuntimeError, "invalid_fmap_format_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_weight_format_case1, RuntimeError, "invalid_weight_format_case1", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_stride_d_case, RuntimeError, "invalid_stride_d_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_stride_h_case, RuntimeError, "invalid_stride_h_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_stride_w_case, RuntimeError, "invalid_stride_w_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_weight_d_case, RuntimeError, "invalid_weight_d_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_weight_h_case, RuntimeError, "invalid_weight_h_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_fmap_w_case, RuntimeError, "invalid_fmap_w_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_fmap_h_case, RuntimeError, "invalid_fmap_h_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_weihgt_w_case, RuntimeError, "invalid_weihgt_w_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(bias_success_case, "success", "Conv3D_default_bias", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(fmap_format_wrong_case, RuntimeError, "fmap_format_wrong", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_weight_format_case2, RuntimeError, "weight_format_wrong", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_stride_length_case, RuntimeError, "wrong_strides", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_dilations_length_case, RuntimeError, "wrong_dilation", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_groups_case, RuntimeError, "wrong_groups", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_dilation_d_case, RuntimeError, "invalid_dilation_d_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_dilation_d_zero_case, RuntimeError, "invalid_dilation_d_zero_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_dilation_h_case, RuntimeError, "invalid_dilation_h_case", True))
        
ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_dilation_w_case, RuntimeError, "invalid_dilation_w_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_format_case, RuntimeError, "invalid_format_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(test_fmap_filter_relation_case1, RuntimeError, "test_fmap_after_pad_d_smaller_than_filter_d", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(test_fmap_filter_relation_case2, RuntimeError, "test_fmap_after_pad_h_smaller_than_filter_h", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(test_fmap_filter_relation_case3, RuntimeError, "test_fmap_after_pad_w_smaller_than_filter_w", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_pads_d_range_case, RuntimeError, "invalid_pads_d_range_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_pad_d_case, RuntimeError, "invalid_pad_d_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_pad_h_range_case, RuntimeError, "invalid_pad_h_range_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_pad_h_case, RuntimeError, "invalid_pad_h_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(invalid_pad_w_range_case, RuntimeError, "invalid_pad_w_range_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(load3D_padding_case, "success", "load3D_padding_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(default_tiling_case, "success", "default_tiling_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(large_batch_default_tiling_case, "success", "large_batch_default_tiling_case", True))

ut_case.add_case(["Ascend310"],
                 _gen_data_case(load2d_case, "success", "load2d_case", True))

ut_case.add_case(["Ascend910A"],
                 _gen_data_case(cycle_buffer_case, "success", "cycle_buffer_case", True))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)


