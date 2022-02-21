#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")


def test_conv2d_aipp_maxpool_fusion(test_arg):
    from tbe.common.context import op_context
    import json
    import te
    from te import tvm
    import te.lang.cce
    from tbe.dsl import auto_schedule
    from te.platform import cce_conf
    from impl.conv2d import conv2d_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.pooling import pool_fuse_compute
    from impl.aipp import aipp_compute
    from impl.fix_pipe import fixpipe_compute
    from impl.max_pool import max_pool_fuse_compute

    def int_ceil_div(num_a, num_b):
        """
        upper division
        """
        if num_b == 0:
            raise RuntimeError(" division by zero")
        return (num_a + num_b - 1) // num_b

    def aipp_conv_relu_maxpooling_template( shape_in_ori, shape_w_ori, \
        in_dtype, w_dtype, strides, pads, dilations, pooling_window, \
        pooling_stride, pooling_padding, bias=None, C04=True,aipp_format="yuv", \
        crops=False, soc_version="Ascend310"):
        N, C, H, W = shape_in_ori

        if soc_version in ["Ascend610", "Ascend710", "Ascend320"] and C04:
            shape_in = (N, (C + 3) // 4, H, W, 4)
        else:
            shape_in = (N, (C + 15) // 16, H, W, 16)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15) // 16
        Co0 = 16

        if C04:
            Ci1 = 1
            Ci0 = 4
            hwc1_4 = int_ceil_div(Hk * Wk * Ci1, 4)
            shape_w = (hwc1_4, Co1, Co0, Ci0 * 4)
            weight_format = "FRACTAL_Z_C04"
        else:
            Ci1 = 1
            Ci0 = 16
            shape_w = (Hk * Wk * Ci1, Co1, Co0, Ci0)
            weight_format = "FRACTAL_Z"

        if aipp_format == "yuv":
            aipp_input_format = "YUV420SP_U8"
        else:
            aipp_input_format = "RGB888_U8"

        with tvm.target.cce():

            data = tvm.placeholder(shape_in_ori,
                                   name="params_0",
                                   dtype="uint8",
                                   attrs={
                                       "ori_shape": shape_in_ori,
                                       "format": "NCHW",
                                       "ori_format": "NHWC"
                                   })

            if crops:
                h_after_crop = shape_in_ori[2] - 4
                w_after_crop = shape_in_ori[3] - 4
            else:
                h_after_crop = shape_in_ori[2]
                w_after_crop = shape_in_ori[3]
            if soc_version in ["Ascend610", "Ascend710", "Ascend320"] and C04:

                output_data = {
                    'shape': [N, 1, h_after_crop, w_after_crop, 4],
                    'ori_shape': [N, 3, h_after_crop, w_after_crop],
                    'format': 'NC1HWC0_C04',
                    'ori_format': 'NCHW',
                    'dtype': 'float16',
                    'addr_type': 0,
                    'valid_shape': (),
                    'slice_offset': (),
                    'use_L1_workspace': 0,
                    'L1_workspace_size': -1,
                    'L1_fusion_type': -1,
                    'L1_addr_offset': 0,
                    'total_shape': (),
                    'split_index': 0
                }
            else:
                output_data = {
                    'shape': [N, 1, h_after_crop, w_after_crop, 16],
                    'ori_shape': [N, 3, h_after_crop, w_after_crop],
                    'format': 'NC1HWC0',
                    'ori_format': 'NCHW',
                    'dtype': 'float16',
                    'addr_type': 0,
                    'valid_shape': (),
                    'slice_offset': (),
                    'use_L1_workspace': 0,
                    'L1_workspace_size': -1,
                    'L1_fusion_type': -1,
                    'L1_addr_offset': 0,
                    'total_shape': (),
                    'split_index': 0
                }

            aipp_config_dict = {
                "cce_product": "2.1",
                "out_dtype": "float16",
                "out_format": "NC1HWC0",
                "aipp_mode": "static",
                "related_input_rank": 0,
                "input_format": aipp_input_format,
                "src_image_size_n": shape_in_ori[0],
                "src_image_size_c": shape_in_ori[1],
                "src_image_size_h": shape_in_ori[2],
                "src_image_size_w": shape_in_ori[3],
                "crop": crops,
                "load_start_pos_h": 2,
                "load_start_pos_w": 2,
                "crop_size_h": h_after_crop,
                "crop_size_w": w_after_crop,
                "src_image_h": shape_in_ori[2],
                "src_image_w": shape_in_ori[3],
                "resize": 0,
                "resize_model": 0,
                "resize_output_h": 32,
                "resize_output_w": 32,
                "padding": 0,
                "left_padding_size": 7,
                "right_padding_size": 7,
                "top_padding_size": 0,
                "bottom_padding_size": 4,
                "csc_switch": 1,
                "rbuv_swap_switch": 0,
                "ax_swap_switch": 0,
                "matrix_r0c0": 298,
                "matrix_r0c1": 516,
                "matrix_r0c2": 0,
                "matrix_r1c0": 298,
                "matrix_r1c1": -100,
                "matrix_r1c2": -208,
                "matrix_r2c0": 298,
                "matrix_r2c1": 0,
                "matrix_r2c2": 409,
                "input_bias_0": 16,
                "input_bias_1": 128,
                "input_bias_2": 128,
                "mean_chn_0": 1,
                "mean_chn_1": 1,
                "mean_chn_2": 1,
                "mean_chn_3": 1,
                "var_reci_chn_0": 1.0,
                "var_reci_chn_1": 1.0,
                "var_reci_chn_2": 1.0,
                "min_chn_0": 0,
                "min_chn_1": 0,
                "min_chn_2": 0,
                "min_chn_3": 0
            }

            aipp_config_dict_json = json.dumps(aipp_config_dict)
            aipp_res = aipp_compute(data,
                                    None,
                                    output_data,
                                    aipp_config_dict_json,
                                    kernel_name="aipp")

            Weight = tvm.placeholder(shape_w,
                                     name='params_1',
                                     dtype=w_dtype,
                                     attrs={
                                         "ori_shape": shape_w_ori,
                                         "ori_format": "NCHW",
                                         "format": weight_format
                                     })  # [13,4,16,16]

            bias_tensor = None

            conv_res = conv2d_compute(aipp_res, Weight, bias_tensor, None,
                                      None, strides, pads, dilations)

            if soc_version == "Ascend320":
                x1 = conv_res
                x2 = None
                quant_scale_0 = None
                relu_weight_0 = None
                clip_value_0 = None
                quant_scale_1 = None
                relu_weight_1 = None
                clip_value_1 = None
                anti_quant_scale = None
                anti_quant_offset = None
                output = {
                    "shape": [8, 4, 112, 112, 16],
                    "format": "NC1HWC0",
                    "dtype": "float16"
                }
                fusion_op_list = []
                unit_list = ["pre_act"]
                eltwise_mode = ""
                relu = fixpipe_compute(
                    conv_res, x2, quant_scale_0, relu_weight_0,
                    clip_value_0, quant_scale_1, relu_weight_1,
                    clip_value_1, anti_quant_scale, anti_quant_offset,
                    output, fusion_op_list, unit_list, eltwise_mode)
            else:
                relu = leaky_relu_compute(conv_res, None, negative_slope=0)
            out = pool_fuse_compute(relu,
                                    None,
                                    None,
                                    None,
                                    pooling_window,
                                    pooling_stride,
                                    pad=pooling_padding)
            sch = auto_schedule(out)

    def aipp_conv_relu_maxpooling_test_cases1(soc):

        shape_in_ori = (8, 3, 224, 224)  #  NC1HWC0
        shape_w_ori = (64, 3, 7, 7)  #  HkWkCi1, Co1, 16, 16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 2, 2]
        pads = [3, 3, 3, 3]
        dilations = [1, 1, 1, 1]
        bias = None
        pooling_window = (3, 3)
        pooling_stride = (2, 2)
        pooling_padding = [0, 1, 0, 1]
        aipp_conv_relu_maxpooling_template( shape_in_ori, shape_w_ori, \
            in_dtype, w_dtype, strides, pads, \
            dilations, pooling_window, pooling_stride, \
            pooling_padding, bias=bias, C04=True, aipp_format="rgb", \
            crops=False, soc_version=soc)

    def aipp_conv_relu_maxpooling_test_cases2():
        with tvm.target.cce():
            aipp_input_data = tvm.placeholder(
                (1, 3, 224, 224),
                name="fmap",
                dtype="uint8",
                attrs={
                    "format": "NCHW",
                    "ori_shape": [1, 3, 224, 224],
                    "ori_format": "NCHW"
                })
            input_dync_param = None
            output_data = {
                "dtype": "float16",
                "format": "NC1HWC0_C04",
                "ori_format": "NHWC",
                "ori_shape": [1, 224, 224, 3],
                "shape": [1, 1, 224, 224, 4],
                "total_shape": [1, 1, 224, 224, 4]
            }
            aipp_config_json_dict = {
                "aipp_mode": "static",
                "bottom_padding_size": 27,
                "crop_size_h": 224,
                "crop_size_w": 224,
                "input_bias_0": 0,
                "input_bias_1": 128,
                "input_bias_2": 128,
                "input_format": "RGB888_U8",
                "left_padding_size": 32,
                "matrix_r0c0": 77,
                "matrix_r0c1": 150,
                "matrix_r0c2": 29,
                "matrix_r1c0": -43,
                "matrix_r1c1": -85,
                "matrix_r1c2": 128,
                "matrix_r2c0": 128,
                "matrix_r2c1": -107,
                "matrix_r2c2": -21,
                "mean_chn_0": 10,
                "mean_chn_1": 10,
                "mean_chn_2": 10,
                "output_bias_0": 0,
                "output_bias_1": 0,
                "output_bias_2": 0,
                "right_padding_size": 23,
                "src_image_size_h": 224,
                "src_image_size_w": 224,
                "top_padding_size": 32,
                "var_reci_chn_0": 0.1,
                "var_reci_chn_1": 0.1,
                "var_reci_chn_2": 0.1
            }
            aipp_config_json = json.dumps(aipp_config_json_dict)
            inputs = aipp_compute(aipp_input_data,
                                  input_dync_param,
                                  output_data,
                                  aipp_config_json,
                                  kernel_name="aipp")

            weights = tvm.placeholder(
                (13, 4, 16, 16),
                name="weights",
                dtype="float16",
                attrs={
                    "format": "FRACTAL_Z_C04",
                    "ori_shape": [7, 7, 3, 64],
                    "ori_format": "HWCN"
                })
            bias = tvm.placeholder((64, ), name="bias", dtype="float32")
            offset_w = None
            outputs = None
            strides = [1, 2, 2, 1]
            pads = [2, 3, 2, 3]
            dilations = [1, 1, 1, 1]
            conv_res = conv2d_compute(inputs, weights, bias, offset_w,
                                        outputs, strides, pads, dilations, data_format="NHWC")

            x2 = None
            quant_scale_0 = None
            relu_weight_0 = None
            clip_value_0 = None
            quant_scale_1 = None
            relu_weight_1 = None
            clip_value_1 = None
            anti_quant_scale = None
            anti_quant_offset = None
            output = {
                "shape": [1, 4, 112, 112, 16],
                "format": "NC1HWC0",
                "dtype": "float16"
            }
            fusion_op_list = []
            unit_list = ["pre_act"]
            eltwise_mode = ""
            relu = fixpipe_compute(conv_res, x2, quant_scale_0,
                                   relu_weight_0, clip_value_0,
                                   quant_scale_1, relu_weight_1,
                                   clip_value_1, anti_quant_scale,
                                   anti_quant_offset, output,
                                   fusion_op_list, unit_list, eltwise_mode)

        input_data = relu
        output_data = {
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_format": "NHWC",
            "ori_shape": [1, 56, 56, 64],
            "shape": [1, 4, 56, 56, 16],
            "total_shape": [1, 4, 56, 56, 16],
            "valid_shape": []
        }
        ksize = [1, 3, 3, 1]
        strides = [1, 2, 2, 1]
        padding = "SAME"
        data_format = "NHWC"
        pool_res = max_pool_fuse_compute(input_data,
                                         output_data,
                                         ksize,
                                         strides,
                                         padding,
                                         data_format,
                                         kernel_name="max_pool_fuse")
        sch = auto_schedule(pool_res)

    with op_context.OpContext():
        TEST_PLATFORM = ["Ascend320"]
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            aipp_conv_relu_maxpooling_test_cases1(soc)
            aipp_conv_relu_maxpooling_test_cases2()

print("adding Conv2D v300 aipp maxpool fusion testcases")
ut_case.add_cust_test_func('Ascend320', test_func=test_conv2d_aipp_maxpool_fusion)
ut_case.run(['Ascend320'])
