#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_eltwise_transdata(test_arg):
    from impl.conv2d import conv2d_compute
    from impl.trans_data import trans_data_compute
    from impl.eltwise import eltwise_compute
    from tbe.common.context import op_context
    from tbe.dsl import auto_schedule
    from tbe.dsl import build
    from te import tvm
    from te import platform as cce_conf
    import te.lang.cce as tbe
    # case name: ((fm_shape), (weight_shape), (paddings), (strides), (dilations), bias_flag)
    testcase = {
        "conv2d_eltwise_transdata_align_1": ((1, 16, 8, 8), (1, 16, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], False),
        "conv2d_eltwise_transdata_align_2": ((1, 16, 960, 1440), (1, 16, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], False),

        "conv2d_eltwise_transdata_not_align_1": ((1, 16, 7, 7), (1, 16, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], False),
        "conv2d_eltwise_transdata_not_align_2": ((1, 16, 965, 1448), (1, 16, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], False),
    }

    def test_conv2d_eltwise_transdata_compile(fm_shape, filter_shape, pads, strides, dilations, bias_flag, kernel_name):
        Ci0 = 16
        Co0 = 16
        batch, Cin, Hin, Win = fm_shape
        Cout, Cin_k, Hk, Wk = filter_shape
        Ci1 = (Cin + Ci0 - 1)//Ci0
        fm_5hd = (batch, Ci1, Hin, Win, Ci0)
        Co1 = (Cout + Co0 - 1)//Co0
        shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)
        input_dtype = "float16"
        _, _, dilate_h, dilate_w = dilations
        Hk_dilated = (Hk - 1)*dilate_h + 1
        Wk_dilated = (Wk - 1)*dilate_w + 1
        Ho = (Hin - Hk_dilated + pads[0] + pads[1]) // strides[2] + 1
        Wo = (Win - Wk_dilated + pads[2] + pads[3]) // strides[3] + 1
        outshape_nc1hwc0 = (batch, Co1, Ho*Wo, Co0)
        outshape_nchw = (batch, Cout, Ho, Wo)

        with op_context.OpContext():
            fmap = tvm.placeholder(fm_5hd, name='fmap', dtype=input_dtype)
            weight = tvm.placeholder(shape_w_fracz, name='weight', dtype=input_dtype,
                                     attrs={'ori_shape': filter_shape, 'ori_format': "NCHW"})
            bias = tvm.placeholder((Cout,), name='bias', dtype=input_dtype) if bias_flag else None
            conv_out = conv2d_compute(fmap, weight, bias, None, None, strides, pads, dilations, kernel_name=kernel_name)

            input_y = tvm.placeholder(outshape_nc1hwc0, name="input_y", dtype="float16")
            elt_out = tbe.vadd(conv_out, input_y)

            dst = {"shape": outshape_nchw}
            trans_out = trans_data_compute(elt_out, dst, "NC1HWC0", "NCHW")
            out = trans_out

            tensor_list = [fmap, weight, bias, input_y, out] if bias_flag else [fmap, weight, input_y, out]
            with tvm.target.cce():
                sch = auto_schedule(out)
            config = {"name": kernel_name,
                      "tensor_list": tensor_list}

        # build(sch, config)

    cce_conf.te_set_version('SD3403')
    for key, value in testcase.items():
        print("test begin test_conv2d_eltwise_transdata_compile case:", key)
        test_conv2d_eltwise_transdata_compile(*value, key)
        print("test end test_conv2d_eltwise_transdata_compile case:", key)


print("test_Conv2D_elt_transdata_lhisi_impl running")
ut_case.add_cust_test_func(test_func=test_conv2d_eltwise_transdata)
