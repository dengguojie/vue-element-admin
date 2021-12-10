#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT
from unittest.mock import MagicMock
from unittest.mock import patch

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_hf32(test_arg):
    from impl.conv2d import conv2d_compute
    from tbe.common.context import op_context
    import tbe.common.context.op_info as operator_info
    from tbe.dsl import auto_schedule
    from te import tvm

    vals = {("Intrinsic_fix_pipe_unit_list",): True,
            ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): False,
            ("Intrinsic_mmad", "h322f32"): True,}

    def mock_platform(*args):
        return vals[args]

    # case name: ((fm_shape), (weight_shape), (paddings), (strides), (dilations), bias_flag, dtype, impl_mode)
    testcase = {
        "conv2d_static_enable_hf32_test_1": ((2, 64, 56, 56), (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], False, "float32", "high_performance"),
        "conv2d_static_disable_hf32_test_1": ((2, 64, 56, 56), (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], False, "float16", "high_performance"),
        "conv2d_static_disable_hf32_test_2": ((2, 64, 56, 56), (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], False, "float32", ""),
    }

    def set_impl_mode(impl_mode):
        cube_op_info = operator_info.OpInfo("conv2d", "Conv2D")
        cube_op_info.precision_mode = impl_mode
        op_context.get_context().add_op_info(cube_op_info)

    def compile_conv2d_hf32(fm_shape, filter_shape, pads, strides, dilations, bias_flag, input_dtype, impl_mode, kernel_name):
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
            set_impl_mode(impl_mode)
            with tvm.target.cce():
                fmap = tvm.placeholder(fm_5hd, name='fmap', dtype=input_dtype)
                weight = tvm.placeholder(shape_w_fracz, name='weight', dtype=input_dtype,
                                         attrs={'ori_shape': filter_shape, 'ori_format': "NCHW"})
                bias = tvm.placeholder((Co0*Co1,), name='bias', dtype=input_dtype) if bias_flag else None
                out = conv2d_compute(fmap, weight, bias, None, None, strides, pads, dilations, kernel_name=kernel_name)
                tensor_list = [fmap, weight, bias, out] if bias_flag else [fmap, weight, out]
                sch = auto_schedule(out)

    for key, value in testcase.items():
        print("test conv2d hf32 case:", key)
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=mock_platform)):
            compile_conv2d_hf32(*value, key)


print("test Conv2D HF32 mode running")
ut_case.add_cust_test_func(test_func=test_conv2d_hf32)
