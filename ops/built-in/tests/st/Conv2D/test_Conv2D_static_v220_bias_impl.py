#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.conv2d import conv2d_compute
from tbe.common.context import op_context
from tbe.dsl import auto_schedule
from te import tvm
from unittest.mock import MagicMock
from unittest.mock import patch

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_static_v220_conv2d_bias(test_arg):

    # case name: ((fm_shape), (weight_shape), (paddings), (strides), (dilations), group, bias_flag, dtype)
    testcase = {
        "conv2d_static_conv2d_bias_test_fp16_1": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float16"),
        "conv2d_static_conv2d_bias_test_fp16_2": ((2, 64, 56, 56), (60, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float16"),
        "conv2d_static_conv2d_bias_test_fp16_3": ((2, 64, 56, 56), (32, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float16"),
        "conv2d_static_conv2d_bias_test_fp16_4": ((2, 64, 56, 56), (30, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float16"),
        "conv2d_static_conv2d_bias_test_fp16_5": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bias_test_fp16_6": ((2, 64, 56, 56), (486, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float16"),
        "conv2d_static_conv2d_bias_test_fp16_7": ((2, 64, 28, 28), (32, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float16"),

        "conv2d_static_conv2d_bias_test_bf16_1": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "bfloat16"),
        "conv2d_static_conv2d_bias_test_bf16_2": ((2, 64, 56, 56), (60, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "bfloat16"),
        "conv2d_static_conv2d_bias_test_bf16_3": ((2, 64, 56, 56), (32, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "bfloat16"),
        "conv2d_static_conv2d_bias_test_bf16_4": ((2, 64, 56, 56), (30, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "bfloat16"),
        "conv2d_static_conv2d_bias_test_bf16_5": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "bfloat16"),

        "conv2d_static_conv2d_bias_test_fp32_1": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float32"),
        "conv2d_static_conv2d_bias_test_fp32_2": ((2, 64, 56, 56), (60, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float32"),
        "conv2d_static_conv2d_bias_test_fp32_3": ((2, 64, 56, 56), (32, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float32"),
        "conv2d_static_conv2d_bias_test_fp32_4": ((2, 64, 56, 56), (30, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, True, "float32"),
        "conv2d_static_conv2d_bias_test_fp32_5": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
    }

    def compile_static_v220_conv2d_bias(fm_shape, filter_shape, pads, strides, dilations, groups, bias_flag, input_dtype, kernel_name):
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
            fmap = tvm.placeholder(fm_5hd, name='fmap', dtype=input_dtype)
            weight = tvm.placeholder(shape_w_fracz, name='weight', dtype=input_dtype,
                                        attrs={'ori_shape': filter_shape, 'ori_format': "NCHW"})
            bias_dtype = "float16" if input_dtype == "bfloat16" else input_dtype
            bias = tvm.placeholder((Cout,), name='bias', dtype=bias_dtype) if bias_flag else None
            out = conv2d_compute(fmap, weight, bias, None, None, strides, pads, dilations, groups=groups, kernel_name=kernel_name)
            tensor_list = [fmap, weight, bias, out] if bias_flag else [fmap, weight, out]
            with tvm.target.cce():
                sch = auto_schedule(out)

    for key, value in testcase.items():
        print("test begin compile_static_v220_conv2d_bias case:", key)
        compile_static_v220_conv2d_bias(*value, key)
        print("test end compile_static_v220_conv2d_bias case:", key)


print("test_Conv2D_static_v220_bias_impl running")
ut_case.add_cust_test_func("Ascend920A", test_func=test_static_v220_conv2d_bias)
ut_case.run(['Ascend920A'])
