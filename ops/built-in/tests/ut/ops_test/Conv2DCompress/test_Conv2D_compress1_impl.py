#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_compress_impl2(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from tbe.dsl import auto_schedule
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d_compress import conv2dcompress_compute, conv2dcompress
    from impl.ascend_dequant import ascend_dequant_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")

    def conv2d_compress_test(fmap_shape, filters_shape, bias_flag, strides, pads, dilations, dtype="int8"):
        if dtype == "int8":
            assert fmap_shape[1] == filters_shape[1]
            batch, channel_in, height_in, width_in = fmap_shape
            channel_in0 = 32
            channel_in1 = (channel_in + channel_in0 - 1) // channel_in0
            fmap_shape_5hd = [batch, channel_in1, height_in, width_in, channel_in0]

            channel_out, channel_in, kernel_h, kernel_w = filters_shape
            channel_out0 = 16
            channel_out1 = (channel_out + channel_out0 - 1) // channel_out0
            filters_shape_fz = [channel_in1 * kernel_h * kernel_w, channel_out1, channel_out0, channel_in0]
        else:
            raise RuntimeError("conv2d_compress does not support data type: %s" % dtype)

        _, _, stride_h, stride_w = strides
        _, _, dilation_h, dilation_w = dilations
        pad_top, pad_bottom, pad_left, pad_right = pads
        height_out = (height_in + pad_top + pad_bottom - (kernel_h - 1) * dilation_h - 1) // stride_h + 1
        width_out = (width_in + pad_left + pad_right - (kernel_w - 1) * dilation_w - 1) // stride_w + 1
        output_shape = [batch, channel_out, height_out, width_out]
        output_shape_5hd = [batch, channel_out1, height_out, width_out, channel_out0]

        inputs = {"shape": fmap_shape_5hd, "ori_shape": fmap_shape, "format": "NC1HWC0", "ori_format": "NCHW",
                  "dtype": "int8"}
        weight = {"shape": filters_shape_fz, "ori_shape": filters_shape, "format": "FRACTAL_Z", "ori_format": "NCHW",
                  "dtype": "int8"}
        index = {"shape": [channel_out1 * channel_out0], "ori_shape": [channel_out], "format": "ND", "ori_format": "ND",
                 "dtype": "int8"}
        outputs = {"shape": output_shape_5hd, "ori_shape": output_shape, "format": "NC1HWC0", "ori_format": "NCHW",
                   "dtype": "int32"}
        bias = None
        offset_w = None
        groups = 1
        data_format = "NCHW"
        offset_x = 0
        kernel_name = "conv2d_compress"

        conv2dcompress(inputs, weight, index, bias, offset_w, outputs, strides, pads, dilations, groups,
                       data_format, offset_x, kernel_name)

    def fusion_conv2dcompress_compute(testcases):
        for testcase in testcases:
            fmap_shape = testcase["fmap_shape"]
            filters_shape = testcase["filters_shape"]
            dtype = testcase["dtype"]
            bias_flag = testcase["bias_flag"]
            pads = testcase["pads"]
            strides = testcase["strides"]
            dilations = testcase["dilations"]

            conv2d_compress_test(fmap_shape, filters_shape, bias_flag,
                                 strides, pads, dilations, dtype)
    testcases = [
        {"fmap_shape": [1, 16, 100, 100], "filters_shape": [32, 16, 3, 3], "dtype": "int8", "bias_flag": None, "pads": [1, 1, 1, 1], "strides": [1, 1, 1, 1], "dilations": [1, 1, 1, 1]},
    ]
    fusion_conv2dcompress_compute(testcases)

    cce_conf.te_set_version(soc_version)

print("test_conv2d_compress_impl2")
ut_case.add_cust_test_func(test_func=test_conv2d_compress_impl2)

if __name__ == "__main__":
    ut_case.add_cust_test_func(test_func=test_conv2d_compress_impl2)
    ut_case.run(["Hi3796CV300CS"])
    exit(0)
