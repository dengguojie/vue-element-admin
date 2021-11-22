#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_transdata(test_arg):
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from tbe.dsl import auto_schedule
    from te import platform as cce_conf
    from impl.conv2d import conv2d_compute
    from impl.conv2d_data_rm import conv2d_data_rm_compute
    from impl.trans_data import trans_data_compute
    from tbe.dsl.compute.conv_compute import ConvParam

    cce_conf.te_set_version('SD3403')
    shape_in = (1, 16, 16, 16)
    shape_w = (1, 16, 3, 3)
    pads = (1, 1, 1, 1)
    strides = (1, 1)

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci1 = (Ci + 15) // 16
    Ci0 = 16

    Co1 = 1
    Co0 = 16

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)
    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    with tvm.target.cce():
        fm = tvm.placeholder(shape_in_5HD, name='fmap', dtype="float16", attrs={'ori_format': 'NCHW'})
        filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype="float16",
                                attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
        bias_tensor = None
        conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=0, options={"invalid_data_rm": True})
        shape_5d = [1, 1, 256]
        trans_data_output = {"shape": shape_5d, "dtype": "float16"}
        trans_out = trans_data_compute(conv_res, trans_data_output, "NC1HWC0", "NCHW")
        out = conv2d_data_rm_compute(trans_out, res_tensor=None)
        tensor_list = [fm, filter_w, out]
        tiling = {'AL0_matrix':[1, 1, 16, 16], 'CL0_matrix': [1, 1, 16, 16, 1], 'CUB_matrix': [1, 1, 16, 16], 
                    'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0, 'BL0_matrix': [1, 1, 16, 16],
                    'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 1, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1, 
                    'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 1, 'CUB_pbuffer': 1, 'UBG_pbuffer': 1},
                    'n_bef_batch_flag': 0, 'AL1_shape': [], 'BL1_shape': None, 'block_dim': [1, 1, 1, 1], 'CUB_channel_wise_flag': False}
        ConvParam.tiling = tiling
        sch = auto_schedule(out)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d",
        "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)

print("adding Conv2D transdata ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_transdata)
