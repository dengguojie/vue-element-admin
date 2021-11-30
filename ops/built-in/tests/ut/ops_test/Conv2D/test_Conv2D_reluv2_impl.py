#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_reluv2_fusion(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from tbe.dsl import auto_schedule
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.relu_v2 import relu_v2_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    cce_conf.te_set_version("Ascend910")

    def conv2d_compute_fusion(fmap_shape, filters_shape, bias_flag, strides, pads, dilations, dtype="float16"):
        if dtype == "float16":
            assert fmap_shape[1] == filters_shape[1]
            ori_fmap_shape = fmap_shape
            ori_fmap_format = "NCHW"
            batch, channel_in, height_in, width_in = fmap_shape
            channel_in0 = 16
            channel_in1 = (channel_in + channel_in0 - 1) // channel_in0
            fmap_shape_5hd = [batch, channel_in1, height_in, width_in, channel_in0]

            ori_filters_shape = filters_shape
            ori_filters_format = "NCHW"
            channel_out, channel_in, kernel_h, kernel_w = filters_shape
            channel_out0 = 16
            channel_out1 = (channel_out + channel_out0 - 1) // channel_out0
            filters_shape_fz = [channel_in1 * kernel_h * kernel_w, channel_out1, channel_out0, channel_in0]
        else:
            raise RuntimeError("conv2d+reluv2 fusion not support data type: %s" %dtype)

        _, _, stride_h, stride_w = strides
        _, _, dilation_h, dilation_w = dilations
        pad_top, pad_bottom, pad_left, pad_right = pads
        height_out = (height_in + pad_top + pad_bottom - (kernel_h - 1) * dilation_h - 1) // stride_h + 1
        width_out = (width_in + pad_left + pad_right - (kernel_w - 1) * dilation_w - 1) // stride_w + 1
        output_shape = [batch, channel_out1, height_out * width_out, channel_out0]

        input_tensors = []

        fmap_data = tvm.placeholder(fmap_shape_5hd, name="fmap_data", dtype=dtype,
                                    attrs={"ori_shape": ori_fmap_shape,
                                           "ori_format": ori_fmap_format})
        filters_data = tvm.placeholder(filters_shape_fz, name="filters_data", dtype=dtype,
                                       attrs={"ori_shape": ori_filters_shape,
                                              "ori_format": ori_filters_format})
        if bias_flag:
            bias_data = tvm.placeholder([channel_out1*channel_out0], name="bias_data",
                                        dtype=dtype if dtype == "floa16" else "int32")
        else:
            bias_data = None
        input_tensors.append(fmap_data)
        input_tensors.append(filters_data)

        conv_res = conv2d_compute(fmap_data, filters_data, bias_data, None, None, strides, pads, dilations)
        return conv_res, input_tensors

    def reluv2_compute_fusion(x):
        reluv2_res = relu_v2_compute(x, None, None)
        return reluv2_res

    def fusion_conv2d_reluv2_compute(testcases):
        for testcase in testcases:
            fmap_shape = testcase["fmap_shape"]
            filters_shape = testcase["filters_shape"]
            dtype = testcase["dtype"]
            bias_flag = testcase["bias_flag"]
            pads = testcase["pads"]
            strides = testcase["strides"]
            dilations = testcase["dilations"]

            total_input_tensors = []
            conv_res, input_tensors = conv2d_compute_fusion(fmap_shape, filters_shape, bias_flag,
                                                            strides, pads, dilations, dtype)
            total_input_tensors += input_tensors
            relu_res = reluv2_compute_fusion(conv_res)
            [total_input_tensors.append(output) for output in relu_res]
            res = [output for output in relu_res]

            config = {"print_ir": False, "need_build": True,
                      "name": "conv2d_reluv2_fusion", "tensor_list": total_input_tensors}
            with tvm.target.cce():
                sch = auto_schedule(res)
            te.lang.cce.cce_build_code(sch, config)
    testcases = [
        {"fmap_shape": [1, 16, 100, 100], "filters_shape": [32, 16, 3, 3], "dtype": "float16", "bias_flag": None, "pads": [1, 1, 1, 1], "strides": [1, 1, 1, 1], "dilations": [1, 1, 1, 1]},
        {"fmap_shape": [1, 64, 300, 450], "filters_shape": [128, 64, 3, 3], "dtype": "float16", "bias_flag": None, "pads": [1, 1, 1, 1], "strides": [1, 1, 1, 1], "dilations": [1, 1, 1, 1]},
    ]
    fusion_conv2d_reluv2_compute(testcases)

print("add Conv2d reluv2 fusion ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_reluv2_fusion)






