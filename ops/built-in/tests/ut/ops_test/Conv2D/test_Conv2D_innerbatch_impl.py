#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT
from unittest.mock import MagicMock
from unittest.mock import patch

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_innerbatch(test_arg):
    from impl.conv2d import conv2d_compute
    from tbe.common.context import op_context
    from tbe.dsl import auto_schedule
    from te import tvm

    def tiling_mock_fp16(*args):
        tiling = {'AL0_matrix': [4, 16, 16, 16, 1, 1], 'AL1_shape': [], 'AUB_channel_wise_flag': None, 'AUB_shape': None,
                  'A_overhead_opt_flag': 0, 'BL0_matrix': [16, 1, 16, 16, 1, 1], 'BL1_shape': None, 'BUB_channel_wise_flag': None,
                  'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [1, 4, 16, 16, 4, 1], 'CUB_channel_wise_flag': False,
                  'CUB_matrix': [1, 4, 16, 16, 4, 1], 'batch_bef_group_flag': 0, 'block_dim': [1, 8, 1, 1],
                  'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 1, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2, 'BL1_pbuffer': 2,
                                             'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 1, 'UBG_pbuffer': 1},
                  'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}
        return tiling

    def tiling_mock_fp32(*args):
        tiling = {'AL0_matrix': [4, 16, 16, 8, 1, 1], 'AL1_shape': [], 'AUB_channel_wise_flag': None, 'AUB_shape': None,
                  'A_overhead_opt_flag': 0, 'BL0_matrix': [16, 1, 16, 8, 1, 1], 'BL1_shape': None, 'BUB_channel_wise_flag': None,
                  'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [1, 4, 16, 16, 4, 1], 'CUB_channel_wise_flag': False,
                  'CUB_matrix': [1, 4, 16, 16, 4, 1], 'batch_bef_group_flag': 0, 'block_dim': [1, 8, 1, 1],
                  'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 1, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2, 'BL1_pbuffer': 2,
                                             'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 1, 'UBG_pbuffer': 1},
                  'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}
        return tiling

    # case name: ((fm_shape), (weight_shape), (paddings), (strides), (dilations), group, bias_flag, dtype, mock)
    testcase = {
        "conv2d_innerbatch_fp16_test_1": ((4, 512, 7, 7), (2048, 512, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16", tiling_mock_fp16),
        "conv2d_innerbatch_fp32_test_1": ((4, 512, 7, 7), (2048, 512, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32", tiling_mock_fp32),
    }

    def compile_conv2d_innerbatch(fm_shape, filter_shape, pads, strides, dilations, groups, bias_flag, input_dtype, mock_obj, kernel_name):
        Ci0 = 16
        if input_dtype == "float32":
            Ci0 = 8
        Co0 = 16
        batch, Cin, Hin, Win = fm_shape
        Cout, Cin_k, Hk, Wk = filter_shape
        Ci1 = (Cin + Ci0 - 1)//Ci0
        fm_5hd = (batch, Ci1, Hin, Win, Ci0)
        Co1 = (Cout + Co0 - 1)//Co0
        shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)

        with op_context.OpContext():
            with tvm.target.cce():
                fmap = tvm.placeholder(fm_5hd, name='fmap', dtype=input_dtype)
                weight = tvm.placeholder(shape_w_fracz, name='weight', dtype=input_dtype,
                                         attrs={'ori_shape': filter_shape, 'ori_format': "NCHW"})
                bias = tvm.placeholder((Co0*Co1,), name='bias', dtype=input_dtype) if bias_flag else None
                with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=mock_obj)):
                    out = conv2d_compute(fmap, weight, bias, None, None, strides, pads, dilations, kernel_name=kernel_name)
                    tensor_list = [fmap, weight, bias, out] if bias_flag else [fmap, weight, out]
                    sch = auto_schedule(out)

    for key, value in testcase.items():
        print("test conv2d innerbatch case:", key)
        compile_conv2d_innerbatch(*value, key)

print("test Conv2D innerbatch mode running")
ut_case.add_cust_test_func("Ascend920A", test_func=test_conv2d_innerbatch)
ut_case.run(['Ascend920A'])
