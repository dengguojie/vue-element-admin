#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_c04_impl(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from topi import generic
    from topi.cce import util
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d
    from impl.util import util_conv2d
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    # cce_conf.te_set_version(["Hi3796CV300ES", "Hi3796CV300CS"])

    def conv2d_c04_test(fmap_shape, filters_shape, bias_flag, strides, pads, dilations, dtype="float16"):
        if dtype == "float16":
            assert fmap_shape[1] == filters_shape[1]
            batch, channel_in, height_in, width_in = fmap_shape
            channel_in0 = 4
            channel_in1 = (channel_in + channel_in0 - 1) // channel_in0
            fmap_shape_5hd = [batch, channel_in1, height_in, width_in, channel_in0]

            channel_out, channel_in, kernel_h, kernel_w = filters_shape
            channel_out0 = 16
            channel_in0_align = 16
            channel_out1 = (channel_out + channel_out0 - 1) // channel_out0
            filters_shape_fz = [(channel_in1 * kernel_h * kernel_w + channel_in0 - 1) //
                                channel_in0, channel_out1, channel_out0, channel_in0_align]
        else:
            raise RuntimeError("conv2d_c04 does not support data type: %s" % dtype)

        _, _, stride_h, stride_w = strides
        _, _, dilation_h, dilation_w = dilations
        pad_top, pad_bottom, pad_left, pad_right = pads
        height_out = (height_in + pad_top + pad_bottom - (kernel_h - 1) * dilation_h - 1) // stride_h + 1
        width_out = (width_in + pad_left + pad_right - (kernel_w - 1) * dilation_w - 1) // stride_w + 1
        output_shape = [batch, channel_out, height_out, width_out]
        output_shape_5hd = [batch, channel_out1, height_out, width_out, channel_out0]

        inputs = {"shape": fmap_shape_5hd, "ori_shape": fmap_shape, "format": "NC1HWC0_C04", "ori_format": "NCHW",
                  "dtype": "float16"}
        weight = {"shape": filters_shape_fz, "ori_shape": filters_shape, "format": "FRACTAL_Z_C04", "ori_format": "NCHW",
                  "dtype": "float16"}
        outputs = {"shape": output_shape_5hd, "ori_shape": output_shape, "format": "NC1HWC0", "ori_format": "NCHW",
                   "dtype": "float16"}
        bias = None
        offset_w = None
        groups = 1
        data_format = "NCHW"
        offset_x = 0
        kernel_name = "conv2d_C04"

        conv2d(inputs, weight, bias, offset_w, outputs, strides, pads, dilations, groups,
               data_format, offset_x, kernel_name)

    def fusion_conv2d_c04_compute(testcases):
        for testcase in testcases:
            fmap_shape = testcase["fmap_shape"]
            filters_shape = testcase["filters_shape"]
            dtype = testcase["dtype"]
            bias_flag = testcase["bias_flag"]
            pads = testcase["pads"]
            strides = testcase["strides"]
            dilations = testcase["dilations"]
            util_conv2d._get_minimum_load_L1(fmap_shape, filters_shape, strides, pads, dilations, data_format="NCHW")

            conv2d_c04_test(fmap_shape, filters_shape, bias_flag,
                                 strides, pads, dilations, dtype)


    def v100_c04_single_op():
        inputs = {'shape': (1, 1, 10, 10, 16), 'ori_shape': (1, 1, 10, 10), 'format': 'NC1HWC0', 'sub_format': 0, 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'sgt_slice_shape': (), 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': [1, 1, 10, 10, 16], 'split_index': 0}
        weights = {'shape': (3, 4, 16, 16), 'ori_shape': (56, 1, 3, 3), 'format': 'FRACTAL_Z_C04', 'sub_format': 0, 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'sgt_slice_shape': (), 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': [3, 4, 16, 16], 'split_index': 0}
        bias = {'shape': (56,), 'ori_shape': (56,), 'format': 'NCHW', 'sub_format': 0, 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'sgt_slice_shape': (), 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': [56], 'split_index': 0}
        offset_w = None
        outputs = {'shape': (1, 4, 10, 10, 16), 'ori_shape': (1, 56, 10, 10), 'format': 'NC1HWC0', 'sub_format': 0, 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'sgt_slice_shape': (), 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': [1, 4, 10, 10, 16], 'split_index': 0}

        strides = (1, 1, 1, 1)
        pads = (1, 1, 1, 1)
        dilations = (1, 1, 1, 1)

        cce_conf.te_set_version('Ascend310')
        conv2d(inputs,
               weights,
               bias,
               offset_w,
               outputs,
               strides,
               pads,
               dilations,
               groups=1,
               offset_x=0,
               kernel_name='conv2d')

    testcases = [
        {"fmap_shape": [1, 4, 1025, 2049], "filters_shape": [32, 4, 3, 3], "dtype": "float16", "bias_flag": None, "pads": [1, 1, 1, 1], "strides": [1, 1, 2, 2], "dilations": [1, 1, 1, 1]},
    ]
    fusion_conv2d_c04_compute(testcases)

    v100_c04_single_op()

print("test_conv2d_c04_impl")
ut_case.add_cust_test_func(test_func=test_conv2d_c04_impl)

if __name__ == "__main__":
    ut_case.add_cust_test_func(test_func=test_conv2d_c04_impl)
    ut_case.run(["Ascend710"])
    exit(0)
