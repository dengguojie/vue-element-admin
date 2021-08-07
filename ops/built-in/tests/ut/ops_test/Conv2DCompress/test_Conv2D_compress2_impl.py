#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_compress_impl1(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from topi import generic
    from topi.cce import util
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d_compress import conv2dcompress_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")

    def conv2d_compress_compute_fusion(fmap_shape, filters_shape, bias_flag, strides, pads, dilations, dtype="int8"):
        if dtype == "int8":
            assert fmap_shape[1] == filters_shape[1]
            ori_fmap_shape = fmap_shape
            ori_fmap_format = "NCHW"
            batch, channel_in, height_in, width_in = fmap_shape
            channel_in0 = 32
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
        compress_index_shape = tvm.var("compress_index_shape", dtype="int32")
        compress_index = tvm.placeholder([compress_index_shape], dtype=dtype)
        input_tensors.append(fmap_data)
        input_tensors.append(filters_data)
        input_tensors.append(compress_index)
        if bias_flag:
            bias_data = tvm.placeholder([channel_out1*channel_out0], name="bias_data",
                                        dtype=dtype if dtype == "floa16" else "int32")
            input_tensors.append(bias_data)
        else:
            bias_data = None

        conv_res = conv2dcompress_compute(fmap_data, filters_data, compress_index, bias_data, 0, None, strides, pads, dilations)
        return conv_res, input_tensors

    def dequant_compute_fusion(x, sqrt_mode=False, relu_flag=False, platform="v100"):
        if len(x.shape) == 5:
            channel_in1, channel_in0 = x.shape[1].value, x.shape[4].value
        elif len(x.shape) == 4:
            channel_in1, channel_in0 = x.shape[1].value, x.shape[3].value
        else:
            raise RuntimeError("dequant x shape is not support")
        deqscale_shape = [1, channel_in1, 1, 1, channel_in0]
        deqscale_data = tvm.placeholder(deqscale_shape, name="dequant_data",
                                        dtype="float16" if platform == "v100" else "uint64",
                                        attrs={"ori_shape": [channel_in1*channel_in0]})
        input_tensors = []
        input_tensors.append(deqscale_data)
        dequant_res = ascend_dequant_compute(x, deqscale_data, None, sqrt_mode=sqrt_mode, relu_flag=relu_flag)
        return dequant_res, input_tensors


    def fusion_conv2dcompress_compute(testcases):
        for testcase in testcases:
            fmap_shape = testcase["fmap_shape"]
            filters_shape = testcase["filters_shape"]
            dtype = testcase["dtype"]
            bias_flag = testcase["bias_flag"]
            pads = testcase["pads"]
            strides = testcase["strides"]
            dilations = testcase["dilations"]

            total_input_tensors = []
            conv_res, input_tensors = conv2d_compress_compute_fusion(fmap_shape, filters_shape, bias_flag,
                                                            strides, pads, dilations, dtype)
            total_input_tensors += input_tensors

            dequant_res, input_tensors = dequant_compute_fusion(conv_res, sqrt_mode=False,
                                                                relu_flag=False, platform="v200")
            total_input_tensors += input_tensors

            total_input_tensors.append(dequant_res)
            res = dequant_res

            config = {"print_ir": False, "need_build": True,
                      "name": "conv2d_compress_fusion", "tensor_list": total_input_tensors}
            with tvm.target.cce():
                sch = generic.auto_schedule(res)
            te.lang.cce.cce_build_code(sch, config)
    testcases = [
        {"fmap_shape": [1, 16, 100, 100], "filters_shape": [32, 16, 3, 3], "dtype": "int8", "bias_flag": None, "pads": [1, 1, 1, 1], "strides": [1, 1, 1, 1], "dilations": [1, 1, 1, 1]},
        {"fmap_shape": [1, 16, 100, 100], "filters_shape": [32, 16, 3, 3], "dtype": "int8", "bias_flag": True, "pads": [1, 1, 1, 1], "strides": [1, 1, 1, 1], "dilations": [1, 1, 1, 1]},
    ]
    fusion_conv2dcompress_compute(testcases)

    cce_conf.te_set_version(soc_version)

print("conv2d_compress fusion ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_compress_impl1)

if __name__ == "__main__":
    ut_case.add_cust_test_func(test_func=test_conv2d_compress_impl1)
    ut_case.run(["Hi3796CV300CS"])
    exit(0)
