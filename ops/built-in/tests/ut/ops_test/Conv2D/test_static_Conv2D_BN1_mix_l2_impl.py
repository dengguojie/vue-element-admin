#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from op_test_frame.ut import OpUT
from te import tvm
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.common.context import op_context
from impl.conv2d import conv2d_compute
from impl.bn_training_reduce import bn_training_reduce_compute


ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")


def test_static_conv2d_bn1_mix_l2(test_arg):
    # case name: ((fm_shape), (weight_shape), (paddings), (strides), (dilations), group, bias_flag, dtype)
    testcase = {
        "conv2d_static_conv2d_bn1_test_fp16_1": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_2": ((2, 128, 28, 28), (512, 128, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_3": ((2, 1024, 14, 14), (2048, 1024, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_4": ((2, 512, 7, 7), (2048, 512, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_5": ((2, 512, 28, 28), (256, 512, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_6": ((2, 256, 56, 56), (128, 256, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_7": ((2, 256, 56, 56), (512, 256, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_8": ((2, 256, 28, 28), (256, 256, 3, 3), [0, 1, 0, 1], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_9": ((2, 64, 56, 56), (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_10": ((2, 1024, 14, 14), (512, 1024, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_11": ((2, 64, 56, 56), (256, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_12": ((2, 128, 56, 56), (128, 128, 3, 3), [0, 1, 0, 1], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_13": ((2, 512, 14, 14), (512, 512, 3, 3), [0, 1, 0, 1], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_14": ((2, 256, 14, 14), (1024, 256, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_15": ((2, 3, 224, 224), (64, 3, 7, 7), [2, 3, 2, 3], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_static_conv2d_bn1_test_fp16_16": ((2, 512, 28, 28), (1024, 512, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),

        "conv2d_static_conv2d_bn1_test_fp32_1": ((2, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_2": ((2, 128, 28, 28), (512, 128, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_3": ((2, 1024, 14, 14), (2048, 1024, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_4": ((2, 512, 7, 7), (2048, 512, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_5": ((2, 512, 28, 28), (256, 512, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_6": ((2, 256, 56, 56), (128, 256, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_7": ((2, 256, 56, 56), (512, 256, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_8": ((2, 256, 28, 28), (256, 256, 3, 3), [0, 1, 0, 1], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_9": ((2, 64, 56, 56), (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_10": ((2, 1024, 14, 14), (512, 1024, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_11": ((2, 64, 56, 56), (256, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_12": ((2, 128, 56, 56), (128, 128, 3, 3), [0, 1, 0, 1], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_13": ((2, 512, 14, 14), (512, 512, 3, 3), [0, 1, 0, 1], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_14": ((2, 256, 14, 14), (1024, 256, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_15": ((2, 3, 224, 224), (64, 3, 7, 7), [2, 3, 2, 3], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_static_conv2d_bn1_test_fp32_16": ((2, 512, 28, 28), (1024, 512, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
    }

    def compile_static_conv2d_bn1_mix_l2(fm_shape, filter_shape, pads, strides, dilations, groups, bias_flag, input_dtype, kernel_name):
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
            bias = tvm.placeholder((Co0*Co1,), name='bias', dtype=input_dtype) if bias_flag else None
            conv_out = conv2d_compute(fmap, weight, bias, None, None, strides, pads, dilations, groups=groups, kernel_name=kernel_name)
            axis = [0, 2, 3]
            bn_out = bn_training_reduce_compute(conv_out, {"format": "NC1HWC0"}, None, axis)

            out = [conv_out, bn_out[0], bn_out[1]]
            with tvm.target.cce():
                sch = auto_schedule(out)

            tensor_list = [fmap, weight, bias] if bias_flag else [fmap, weight]

            real_outs = sch.cce_special["real_out_tensor"]
            tensor_list.extend(real_outs)
            print(tensor_list)
            config = {"name": kernel_name,
                      "tensor_list": tensor_list}

        # build(sch, config)

    for key, value in testcase.items():
        print("test begin test_static_conv2d_bn1_mix_l2 case:", key)
        compile_static_conv2d_bn1_mix_l2(*value, key)
        print("test end test_static_conv2d_bn1_mix_l2 case:", key)


print("test_static_conv2d_bn1_mix_l2 running")
ut_case.add_cust_test_func("Ascend920A", test_func=test_static_conv2d_bn1_mix_l2)
ut_case.run(["Ascend920A"])
