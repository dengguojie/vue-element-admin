#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import te.lang.cce
from te import tvm
from te import platform as cce_conf
from tbe.dsl import auto_schedule
from impl.conv2d import conv2d_compute
from impl.relu6 import relu6_compute
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_quant import ascend_quant_compute
from impl.conv2d_data_rm import conv2d_data_rm_compute
from tbe.common.context import op_context
from impl.trans_data import trans_data_compute
from tbe.dsl.compute.conv_compute import ConvParam


def test_conv2d_rmpad():
    cce_conf.te_set_version('Ascend310')
    shape_in = (16, 1024, 7, 7)
    shape_w = (1024, 1024, 1, 1)
    pads = (0, 0, 0, 0)
    strides = (1, 1)

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci1 = (Ci + 31) // 32
    Ci0 = 32

    Co1 = (Co + 15) // 16
    Co0 = 16

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    with tvm.target.cce():
        fm = tvm.placeholder(shape_in_5HD,
                             name='fmap',
                             dtype="int8",
                             attrs={'ori_format': 'NCHW'})

        filter_w = tvm.placeholder(shape_w_fracz,
                                   name='filter_w',
                                   dtype="int8",
                                   attrs={
                                       'ori_shape': shape_w,
                                       'ori_format': 'NCHW'
                                   })
        bias_tensor = None
        vdeq = tvm.placeholder(shape_scale,
                               name='vreq_reg',
                               dtype="float16",
                               attrs={'ori_shape': [Co1 * Co0]})

        conv_res = conv2d_compute(fm,
                                  filter_w,
                                  bias_tensor,
                                  None,
                                  None,
                                  strides,
                                  pads,
                                  dilations,
                                  offset_x=0,
                                  options={"invalid_data_rm": True})
        dequant = ascend_dequant_compute(conv_res,
                                         vdeq,
                                         None,
                                         sqrt_mode=False,
                                         relu_flag=False)
        relu = relu6_compute(dequant, None)
        out = ascend_quant_compute(relu,
                                   None,
                                   scale=1,
                                   offset=0,
                                   sqrt_mode=False)
        out = conv2d_data_rm_compute(out, res_tensor=None)
        tensor_list = [fm, filter_w, vdeq, out]
        sch = auto_schedule(out)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d",
        "tensor_list": tensor_list
    }
    te.lang.cce.cce_build_code(sch, config)


def test_conv2d_transdata():
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
    shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)
    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    with tvm.target.cce():
        fm = tvm.placeholder(shape_in_5HD,
                             name='fmap',
                             dtype="float16",
                             attrs={'ori_format': 'NCHW'})
        filter_w = tvm.placeholder(shape_w_fracz,
                                   name='filter_w',
                                   dtype="float16",
                                   attrs={
                                       'ori_shape': shape_w,
                                       'ori_format': 'NCHW'
                                   })
        bias_tensor = None
        conv_res = conv2d_compute(fm,
                                  filter_w,
                                  bias_tensor,
                                  None,
                                  None,
                                  strides,
                                  pads,
                                  dilations,
                                  offset_x=0,
                                  options={"invalid_data_rm": True})
        shape_5d = [1, 1, 256]
        trans_data_output = {"shape": shape_5d, "dtype": "float16"}
        trans_out = trans_data_compute(conv_res, trans_data_output, "NC1HWC0",
                                       "NCHW")
        out = conv2d_data_rm_compute(trans_out, res_tensor=None)
        tensor_list = [fm, filter_w, out]
        tiling = {
            'AL0_matrix': [1, 1, 16, 16],
            'CL0_matrix': [1, 1, 16, 16, 1],
            'CUB_matrix': [1, 1, 16, 16],
            'A_overhead_opt_flag': 0,
            'B_overhead_opt_flag': 0,
            'BL0_matrix': [1, 1, 16, 16],
            'manual_pingpong_buffer': {
                'AL0_pbuffer': 1,
                'AL1_pbuffer': 1,
                'AUB_pbuffer': 1,
                'BL0_pbuffer': 1,
                'BL1_pbuffer': 1,
                'BUB_pbuffer': 1,
                'CL0_pbuffer': 1,
                'CUB_pbuffer': 1,
                'UBG_pbuffer': 1
            },
            'n_bef_batch_flag': 0,
            'AL1_shape': [],
            'BL1_shape': None,
            'block_dim': [1, 1, 1, 1],
            'CUB_channel_wise_flag': False
        }
        ConvParam.tiling = tiling
        sch = auto_schedule(out)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d",
        "tensor_list": tensor_list
    }
    te.lang.cce.cce_build_code(sch, config)


if __name__ == '__main__':
    with op_context.OpContext():
        test_conv2d_rmpad()
        test_conv2d_transdata()
