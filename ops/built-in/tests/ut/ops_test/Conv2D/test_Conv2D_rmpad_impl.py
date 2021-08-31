#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_rmpad(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from tbe.dsl import auto_schedule
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.relu6 import relu6_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.conv2d_data_rm import conv2d_data_rm_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

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
    shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    with tvm.target.cce():
        fm = tvm.placeholder(shape_in_5HD, name='fmap', dtype="int8", attrs={'ori_format': 'NCHW'})

        filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype="int8",
                                   attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
        bias_tensor = None
        vdeq = tvm.placeholder(shape_scale, name='vreq_reg', dtype="float16",
                               attrs={'ori_shape': [Co1*Co0]})

        conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=0, options={"invalid_data_rm": True})
        dequant = ascend_dequant_compute(conv_res, vdeq, None, sqrt_mode=False, relu_flag=False)
        relu = relu6_compute(dequant, None)
        out = ascend_quant_compute(relu, None, scale=1, offset=0, sqrt_mode=False)
        out = conv2d_data_rm_compute(out, res_tensor=None)
        tensor_list = [fm, filter_w, vdeq, out]
        sch = auto_schedule(out)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d",
        "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)

print("adding Conv2D rmpad ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_rmpad)
