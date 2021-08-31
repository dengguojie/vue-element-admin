#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")
def test_tiling_query(test_arg):

    import numpy as np
    from te import tvm
    from te import platform as cce
    from te.platform import scope_ubuf
    from tbe.common.tiling.tiling_query import tiling_query

    # the shape not in repository
    img_shape_predict = [1, 1, 224, 224, 16]
    filter_shape_predict = [1, 1, 224, 224, 16]
    out_shape_repo = [1, 1, 1, 1, 16]
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float32'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 1, 1
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_filter'

    tiling_predict = {'BL1_shape': [7168, 1, 1], 'A_overhead_opt_flag': 0,
                        'manual_pingpong_buffer': {'AUB_pbuffer': 1, 'AL0_pbuffer': 2,
                        'UBG_pbuffer': 1, 'AL1_pbuffer': 2, 'CL0_pbuffer': 2,
                        'BUB_pbuffer': 1, 'BL1_pbuffer': 2, 'BL0_pbuffer': 2,
                        'CUB_pbuffer': 2}, 'block_dim': [1, 1, 1], 'CUB_matrix': [1, 1, 16, 16, 1],
                        'AUB_shape': None, 'AL1_shape': [7168, 1, 1], 'B_overhead_opt_flag': 0,
                        'BUB_shape': None, 'BL0_matrix': [64, 1, 16, 16, 1], 'AL0_matrix': [1, 64, 16, 16, 1],
                        'CL0_matrix': [1, 1, 16, 16, 1], 'cout_bef_batch_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_repository_query_003\n")

    # the shape not in repository
    img_shape_predict = [16, 32, 19, 19, 16]
    filter_shape_predict = [512, 8, 1, 1, 16]
    out_shape_repo = [16, 8, 56, 56, 16]
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float32'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 3, 3
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'BL1_shape': [], 'BUB_shape': None,
                        'CL0_matrix': [4, 23, 16, 16, 1], 'AL0_matrix': [23, 2, 16, 16, 1], 'AL1_shape': [],
                        'manual_pingpong_buffer': {'AUB_pbuffer': 1, 'AL0_pbuffer': 2, 'BL0_pbuffer': 2,
                        'AL1_pbuffer': 1, 'BUB_pbuffer': 1, 'UBG_pbuffer': 1, 'BL1_pbuffer': 1,
                        'CUB_pbuffer': 1, 'CL0_pbuffer': 2}, 'cout_bef_batch_flag': 0,
                            'B_overhead_opt_flag': 0, 'A_overhead_opt_flag': 0, 'block_dim': [2, 1, 1],
                            'AUB_shape': None, 'CUB_matrix': [1, 23, 16, 16, 1], 'BL0_matrix': [2, 4, 16, 16, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)

    print("test_tiling_cost_model_query_004\n")

    # conv2d_backprop_filter_001
    # the shape not in repository
    img_shape_predict = [8, 1, 48, 48, 16]
    filter_shape_predict = [64, 1, 7, 7, 16]
    out_shape_repo = [1, 8, 16, 16, 48]
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float32'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 1, 1
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_filter'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BL1_pbuffer': 2,
                        'AL0_pbuffer': 2, 'BUB_pbuffer': 1, 'BL0_pbuffer': 1,
                        'UBG_pbuffer': 1, 'CL0_pbuffer': 2, 'AUB_pbuffer': 1,
                        'AL1_pbuffer': 2}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [128, 1, 16, 16, 1], 'AL1_shape': [2304, 1, 1],
                        'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [128, 1, 16, 16, 1],
                        'AUB_shape': None, 'cout_bef_batch_flag': 0, 'BL0_matrix': [1, 128, 16, 16, 1],
                        'AL0_matrix': [1, 1, 16, 16, 1], 'BL1_shape': [2304, 1, 1], 'block_dim': [1, 1, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_005\n")

    # conv2d_backprop_filter_002
    # the shape not in repository
    img_shape_predict = [8, 1, 2, 1, 16]
    filter_shape_predict = [13, 16, 1, 1, 16]
    out_shape_repo = [13, 16, 16, 16, 48]
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'int8'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 1, 1
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_filter'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BL1_pbuffer': 2,
                        'AL0_pbuffer': 2, 'BUB_pbuffer': 1, 'BL0_pbuffer': 1, 'UBG_pbuffer': 1,
                        'CL0_pbuffer': 2, 'AUB_pbuffer': 1, 'AL1_pbuffer': 2}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [128, 1, 16, 16, 1], 'AL1_shape': [16, 1, 1], 'BUB_shape': None,
                        'B_overhead_opt_flag': 0, 'CL0_matrix': [128, 1, 16, 16, 1], 'AUB_shape': None,
                        'cout_bef_batch_flag': 0, 'BL0_matrix': [1, 128, 16, 16, 1],
                        'AL0_matrix': [1, 1, 16, 16, 1], 'BL1_shape': [16, 1, 1], 'block_dim': [1, 1, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_006\n")

    # conv2d_backprop_filter_003
    # the shape not in repository
    img_shape_predict = [2, 1, 2, 1, 16]
    filter_shape_predict = [53, 2, 1, 1, 16]
    out_shape_repo = [1, 16, 16, 16, 48]
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'uint8'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 2, 2
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 1
    op_tag = 'conv2d_backprop_filter'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BL1_pbuffer': 2,
                        'AL0_pbuffer': 2, 'BUB_pbuffer': 1, 'BL0_pbuffer': 1, 'UBG_pbuffer': 1,
                        'CL0_pbuffer': 2, 'AUB_pbuffer': 1, 'AL1_pbuffer': 2}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [128, 1, 16, 16, 1], 'AL1_shape': [16, 1, 1], 'BUB_shape': None,
                        'B_overhead_opt_flag': 0, 'CL0_matrix': [128, 1, 16, 16, 1], 'AUB_shape': None,
                        'cout_bef_batch_flag': 0, 'BL0_matrix': [1, 128, 16, 16, 1],
                        'AL0_matrix': [1, 1, 16, 16, 1], 'BL1_shape': [16, 1, 1], 'block_dim': [1, 1, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_007\n")

    # conv2d_backprop_filter_004
    # the shape not in repository
    img_shape_predict = [2, 1, 65535, 1, 16]
    filter_shape_predict = [27, 1, 1, 1, 16]
    out_shape_repo = [1, 16, 16, 16, 48]
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float16'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 1, 1
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_filter'

    tiling_predict = {'block_dim': [1, 1, 1], 'A_overhead_opt_flag': 0,
                        'BL0_matrix': [1, 128, 16, 16, 1], 'cout_bef_batch_flag': 0,
                        'AUB_shape': None, 'CL0_matrix': [128, 1, 16, 16, 1],
                        'BUB_shape': None, 'AL0_matrix': [1, 1, 16, 16, 1],
                        'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'AL1_pbuffer': 1,
                        'BL0_pbuffer': 1, 'UBG_pbuffer': 1, 'BUB_pbuffer': 1,
                        'AUB_pbuffer': 1, 'AL0_pbuffer': 2, 'CL0_pbuffer': 2,
                        'BL1_pbuffer': 1}, 'BL1_shape': [16384, 1, 1], 'AL1_shape': [16384, 1, 1],
                        'CUB_matrix': [128, 1, 16, 16, 1], 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_008\n")

    # conv2d_backprop_input_001
    # the shape not in repository
    img_shape_predict = [8, 1, 48, 48, 16]
    filter_shape_predict = [16, 1, 7, 7, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float32'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 2, 2
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'AUB_shape': [225792, 2048, 1], 'BL1_shape': [], 'BL0_matrix': [],
                        'block_dim': [2, 1, 1], 'AL0_matrix': [64, 1, 16, 16, 1], 'CL0_matrix': [1, 64, 16, 16, 1],
                        'cout_bef_batch_flag': 0, 'AL1_shape': [], 'A_overhead_opt_flag': 0, 'BUB_shape': None,
                        'CUB_matrix': [1, 64, 16, 16, 1], 'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BUB_pbuffer': 1,
                        'AUB_pbuffer': 2, 'AL1_pbuffer': 2, 'UBG_pbuffer': 2, 'BL0_pbuffer': 1, 'CL0_pbuffer': 2, 'AL0_pbuffer': 2,
                        'BL1_pbuffer': 2}, 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_009\n")

    # conv2d_backprop_input_002
    # the shape not in repository
    img_shape_predict = [8, 16, 2, 1, 16]
    filter_shape_predict = [256, 16, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 2, 2
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2,
                        'BL1_pbuffer': 2, 'AL0_pbuffer': 2, 'BUB_pbuffer': 1,
                        'BL0_pbuffer': 2, 'UBG_pbuffer': 2, 'CL0_pbuffer': 2,
                        'AUB_pbuffer': 2, 'AL1_pbuffer': 2}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [4, 2, 16, 16, 1], 'AL1_shape': [], 'BUB_shape': None,
                        'B_overhead_opt_flag': 0, 'CL0_matrix': [4, 2, 16, 16, 1],
                        'AUB_shape': [2048, 2048, 1], 'cout_bef_batch_flag': 0,
                        'BL0_matrix': [8, 4, 16, 16, 1], 'AL0_matrix': [2, 8, 16, 16, 1],
                        'BL1_shape': [], 'block_dim': [2, 1, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0010\n")

    # conv2d_backprop_input_003
    # the shape not in repository
    img_shape_predict = [4, 2, 4, 2, 16]
    filter_shape_predict = [32, 2, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 3, 3
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 1
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'AUB_shape': [512, 4608, 1], 'BL1_shape': [], 'BL0_matrix': [],
                        'block_dim': [2, 1, 1], 'AL0_matrix': [6, 1, 16, 32, 1],
                        'CL0_matrix': [2, 6, 16, 16, 1], 'cout_bef_batch_flag': 0,
                        'AL1_shape': [], 'A_overhead_opt_flag': 0, 'BUB_shape': None,
                        'CUB_matrix': [2, 6, 16, 16, 1], 'manual_pingpong_buffer': {'CUB_pbuffer': 2,
                        'BUB_pbuffer': 1, 'AUB_pbuffer': 2, 'AL1_pbuffer': 2, 'UBG_pbuffer': 2, 'BL0_pbuffer': 1,
                        'CL0_pbuffer': 2, 'AL0_pbuffer': 2, 'BL1_pbuffer': 2}, 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0011\n")

    # conv2d_backprop_input_004
    # the shape not in repository
    img_shape_predict = [2, 1, 65535, 1, 16]
    filter_shape_predict = [16, 1, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float16'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 5, 5
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'AUB_shape': [816, 12800, 1], 'BL1_shape': [],
                        'BL0_matrix': [], 'block_dim': [2, 1, 1], 'AL0_matrix': [63, 1, 16, 16, 1],
                        'CL0_matrix': [1, 63, 16, 16, 1], 'cout_bef_batch_flag': 0, 'AL1_shape': [16, 19, 1],
                        'A_overhead_opt_flag': 0, 'BUB_shape': None, 'CUB_matrix': [1, 63, 16, 16, 1],
                        'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BUB_pbuffer': 1, 'AUB_pbuffer': 2,
                        'AL1_pbuffer': 1, 'UBG_pbuffer': 2, 'BL0_pbuffer': 1, 'CL0_pbuffer': 2,
                        'AL0_pbuffer': 2, 'BL1_pbuffer': 2}, 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0012\n")

    # conv2d_backprop_input_optimize_001
    # the shape not in repository
    img_shape_predict = [2, 1, 64, 1, 16]
    filter_shape_predict = [16, 1, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float16'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 3, 3
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'AUB_shape': None, 'BL1_shape': [], 'BL0_matrix': [1, 1, 16, 16, 1],
                        'block_dim': [2, 1, 1], 'AL0_matrix': [4, 1, 16, 16, 1], 'CL0_matrix': [1, 4, 16, 16, 1],
                        'cout_bef_batch_flag': 0, 'AL1_shape': [], 'A_overhead_opt_flag': 0, 'BUB_shape': None,
                        'CUB_matrix': [1, 4, 16, 16, 1], 'manual_pingpong_buffer': {'CUB_pbuffer': 2,
                        'BUB_pbuffer': 1, 'AUB_pbuffer': 1, 'AL1_pbuffer': 1, 'UBG_pbuffer': 1, 'BL0_pbuffer': 2,
                        'CL0_pbuffer': 2, 'AL0_pbuffer': 2, 'BL1_pbuffer': 1}, 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0013\n")

    # conv2d_backprop_input_optimize_002
    # the shape not in repository
    img_shape_predict = [4, 4, 8, 2, 16]
    filter_shape_predict = [64, 1, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float16'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 3, 3
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'AUB_shape': None, 'BL1_shape': [], 'BL0_matrix': [4, 1, 16, 16, 1],
                        'block_dim': [2, 1, 1], 'AL0_matrix': [1, 4, 16, 16, 1], 'CL0_matrix': [1, 1, 16, 16, 1],
                        'cout_bef_batch_flag': 0, 'AL1_shape': [], 'A_overhead_opt_flag': 0, 'BUB_shape': None,
                        'CUB_matrix': [1, 1, 16, 16, 1], 'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BUB_pbuffer': 1,
                        'AUB_pbuffer': 1, 'AL1_pbuffer': 1, 'UBG_pbuffer': 1, 'BL0_pbuffer': 2, 'CL0_pbuffer': 2,
                        'AL0_pbuffer': 2, 'BL1_pbuffer': 1}, 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0014\n")

    # conv2d_backprop_input_optimize_003
    # the shape not in repository
    img_shape_predict = [8, 16, 48, 48, 16]
    filter_shape_predict = [256, 7, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float32'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 1, 1
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'AUB_shape': None, 'BL1_shape': [], 'BL0_matrix': [1, 2, 16, 16, 1],
                        'block_dim': [2, 1, 1], 'AL0_matrix': [36, 1, 16, 16, 1],
                        'CL0_matrix': [2, 36, 16, 16, 1], 'cout_bef_batch_flag': 0,
                        'AL1_shape': [16, 1, 1], 'A_overhead_opt_flag': 0, 'BUB_shape': None,
                        'CUB_matrix': [2, 36, 16, 16, 1], 'manual_pingpong_buffer': {'CUB_pbuffer': 2,
                        'BUB_pbuffer': 1, 'AUB_pbuffer': 1, 'AL1_pbuffer': 2, 'UBG_pbuffer': 1,
                        'BL0_pbuffer': 2, 'CL0_pbuffer': 2, 'AL0_pbuffer': 2, 'BL1_pbuffer': 1}, 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0015\n")

    # conv2d_backprop_input_optimize_004
    # the shape not in repository
    img_shape_predict = [2, 1, 64, 1, 16]
    filter_shape_predict = [16, 1, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float32'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 1, 1
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'AUB_shape': None, 'BL1_shape': [], 'BL0_matrix': [1, 1, 16, 16, 1],
                        'block_dim': [2, 1, 1], 'AL0_matrix': [4, 1, 16, 16, 1],
                        'CL0_matrix': [1, 4, 16, 16, 1], 'cout_bef_batch_flag': 0, 'AL1_shape': [],
                        'A_overhead_opt_flag': 0, 'BUB_shape': None, 'CUB_matrix': [1, 4, 16, 16, 1],
                        'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BUB_pbuffer': 1, 'AUB_pbuffer': 1,
                        'AL1_pbuffer': 1, 'UBG_pbuffer': 1, 'BL0_pbuffer': 2, 'CL0_pbuffer': 2,
                        'AL0_pbuffer': 2, 'BL1_pbuffer': 1}, 'B_overhead_opt_flag': 0}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0016\n")

    # conv2d_backprop_input_005
    # the shape not in repository
    img_shape_predict = [256, 16, 48, 48, 16]
    filter_shape_predict = [256, 1, 27, 27, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float32'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 0, 0, 0, 0  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 2, 2
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2,
                        'BL1_pbuffer': 1, 'AL0_pbuffer': 2, 'BUB_pbuffer': 1,
                        'BL0_pbuffer': 2, 'UBG_pbuffer': 2, 'CL0_pbuffer': 2,
                        'AUB_pbuffer': 2, 'AL1_pbuffer': 1}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [1, 64, 16, 16, 1], 'AL1_shape': [23328, 1, 1],
                        'BUB_shape': None, 'B_overhead_opt_flag': 0,
                        'CL0_matrix': [1, 64, 16, 16, 1], 'AUB_shape': [3359232, 2048, 1],
                        'cout_bef_batch_flag': 0, 'BL0_matrix': [1, 1, 16, 16, 1],
                        'AL0_matrix': [64, 1, 16, 16, 1], 'BL1_shape': [23328, 1, 1], 'block_dim': [2, 1, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0017\n")

    # conv2d_backprop_input_006
    # the shape not in repository
    img_shape_predict = [8, 256, 2, 1, 16]
    filter_shape_predict = [4096, 16, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 2, 2
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2,
                        'BL1_pbuffer': 1, 'AL0_pbuffer': 2, 'BUB_pbuffer': 1,
                        'BL0_pbuffer': 2, 'UBG_pbuffer': 2, 'CL0_pbuffer': 2,
                        'AUB_pbuffer': 2, 'AL1_pbuffer': 2}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [4, 2, 16, 16, 1], 'AL1_shape': [], 'BUB_shape': None,
                        'B_overhead_opt_flag': 0, 'CL0_matrix': [4, 2, 16, 16, 1],
                        'AUB_shape': [4096, 2048, 1], 'cout_bef_batch_flag': 0,
                        'BL0_matrix': [8, 4, 16, 16, 1], 'AL0_matrix': [2, 8, 16, 16, 1],
                        'BL1_shape': [4096, 1, 1], 'block_dim': [2, 1, 1]}


    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0018\n")

    # conv2d_backprop_input_007
    # the shape not in repository
    img_shape_predict = [4, 200, 4, 2, 16]
    filter_shape_predict = [3200, 2, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float32'
    padl_predict, padr_predict, padu_predict, padd_predict = 3, 3, 3, 3  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 3, 3
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 1
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2, 'BL1_pbuffer': 2,
                        'AL0_pbuffer': 2, 'BUB_pbuffer': 1, 'BL0_pbuffer': 2, 'UBG_pbuffer': 2,
                        'CL0_pbuffer': 2, 'AUB_pbuffer': 2, 'AL1_pbuffer': 1}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [2, 12, 16, 16, 1], 'AL1_shape': [], 'BUB_shape': None,
                        'B_overhead_opt_flag': 0, 'CL0_matrix': [2, 12, 16, 16, 1],
                        'AUB_shape': [2560, 4608, 1], 'cout_bef_batch_flag': 0,
                        'BL0_matrix': [5, 2, 16, 32, 1], 'AL0_matrix': [12, 5, 16, 32, 1],
                        'BL1_shape': [], 'block_dim': [2, 1, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0019\n")

    # conv2d_backprop_input_008
    # the shape not in repository
    img_shape_predict = [2, 512, 65535, 1, 16]
    filter_shape_predict = [8192, 1, 1, 1, 16]
    out_shape_repo = None
    img_dtype_predict = 'float16'
    filter_dtype_predict = 'float16'
    res_dtype_predict = 'float16'
    mad_type_repo = 'float16'
    padl_predict, padr_predict, padu_predict, padd_predict = 1, 1, 1, 1  # no case
    strideH_predict, strideW_predict = 1, 1
    strideH_expand_repo, strideW_expand_repo = 5, 5
    dilationH_predict, dilationW_predict = 1, 1
    group_predict = 1
    fused_double_operand_num = 0
    bias_flag_predict = 0
    op_tag = 'conv2d_backprop_input'

    tiling_predict = {'manual_pingpong_buffer': {'CUB_pbuffer': 2,
                        'BL1_pbuffer': 1, 'AL0_pbuffer': 2, 'BUB_pbuffer': 1,
                        'BL0_pbuffer': 2, 'UBG_pbuffer': 2, 'CL0_pbuffer': 2,
                        'AUB_pbuffer': 2, 'AL1_pbuffer': 1}, 'A_overhead_opt_flag': 0,
                        'CUB_matrix': [1, 64, 16, 16, 1], 'AL1_shape': [256, 1, 1],
                        'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [1, 64, 16, 16, 1],
                        'AUB_shape': [1088, 12800, 1], 'cout_bef_batch_flag': 0,
                        'BL0_matrix': [1, 1, 16, 16, 1], 'AL0_matrix': [64, 1, 16, 16, 1],
                        'BL1_shape': [], 'block_dim': [2, 1, 1]}

    tiling_res_predict = tiling_query(img_shape_predict, filter_shape_predict,
                                        out_shape_repo, img_dtype_predict,
                                        filter_dtype_predict, res_dtype_predict,
                                        mad_type_repo, padl_predict, padr_predict,
                                        padu_predict, padd_predict, strideH_predict,
                                        strideW_predict, strideH_expand_repo,
                                        strideW_expand_repo, dilationH_predict,
                                        dilationW_predict, group_predict,
                                        fused_double_operand_num, bias_flag_predict,
                                        op_tag);
    print(tiling_res_predict)
    # .assertEquals(tiling_res_predict, tiling_predict)
    print("test_tiling_cost_model_query_0020\n")

print("adding Conv2D tiling query testcases")
ut_case.add_cust_test_func(test_func=test_tiling_query)

