#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv_aipp(test_arg):
    """
    Copyright 2018 Huawei Technologies Co., Ltd
    aipp + conv ut
    """
    import impl
    import topi
    import unittest
    import json
    import te
    from te import tvm
    from topi.cce import util
    import te.lang.cce
    from te.lang.cce import cce_build_code
    from topi import generic
    from te.platform import cce_conf
    from te.platform.fusion_util import fusion_op
    from te.platform.fusion_manager import fusion_manager
    from impl.ascend_quant import ascend_quant_compute
    from impl.conv2d import conv2d_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.pooling import pool_fuse_compute
    from impl.aipp import aipp_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    TEST_PLATFORM = ["Ascend310","Ascend610","Ascend710","Hi3796CV300ES"]

    def int_ceil_div(num_a, num_b):
        """
        upper division
        """
        if num_b == 0:
            raise RuntimeError(" division by zero")
        return (num_a + num_b - 1) // num_b

    def aipp_conv_relu_template( shape_in_ori, shape_w_ori, in_dtype, w_dtype, \
        strides, pads, dilations, bias=None, C04=True,aipp_format="yuv", \
        crops=False, soc_version="Ascend310"):
        N, C, H, W = shape_in_ori

        if soc_version in ["Ascend610", "Ascend710"] and C04:
            shape_in = (N, (C + 3)//4, H, W, 4)
        else:
            shape_in = (N, (C + 15)//16, H, W, 16)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16

        if C04:
            Ci1 = 1
            Ci0 = 4
            hwc1_4 = int_ceil_div(Hk*Wk*Ci1, 4)
            shape_w = (hwc1_4, Co1, Co0, Ci0*4)
            weight_format = "FRACTAL_Z_C04"
        else:
            Ci1 = 1
            Ci0 = 16
            shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)
            weight_format = "FRACTAL_Z"

        if aipp_format == "yuv":
            aipp_input_format = "YUV420SP_U8"
        elif aipp_format == "xrgb":
            aipp_input_format = "XRGB8888_U8"
        else:
            aipp_input_format = "RGB888_U8"


        with tvm.target.cce():

            data = tvm.placeholder(shape_in_ori, name="params_0", dtype="uint8",attrs={"ori_shape": shape_in_ori, "format" : "NCHW", "ori_format": "NHWC"})

            if crops:
                h_after_crop = shape_in_ori[2] - 4
                w_after_crop = shape_in_ori[3] - 4
            else:
                h_after_crop = shape_in_ori[2]
                w_after_crop = shape_in_ori[3]
            if soc_version in ["Ascend610", "Ascend710"] and C04:

                output_data = {'shape': [N,1,h_after_crop,w_after_crop,4], 'ori_shape': [N,3,h_after_crop,w_after_crop], 'format': 'NC1HWC0_C04', 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'use_L1_workspace': 0, 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': (), 'split_index': 0}
            else:
                output_data = {'shape': [N,1,h_after_crop,w_after_crop,16], 'ori_shape': [N,3,h_after_crop,w_after_crop], 'format': 'NC1HWC0', 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'use_L1_workspace': 0, 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': (), 'split_index': 0}

            aipp_config_dict = {"cce_product" : "2.1",
                            "out_dtype" : "float16",
                            "out_format" : "NC1HWC0",
                            "aipp_mode" : "static",
                            "related_input_rank" : 0,
                            "input_format" :aipp_input_format,
                            "src_image_size_n" : shape_in_ori[0],
                            "src_image_size_c" : shape_in_ori[1],
                            "src_image_size_h" : shape_in_ori[2],
                            "src_image_size_w" : shape_in_ori[3],
                            "crop" : crops,
                            "load_start_pos_h" : 2,
                            "load_start_pos_w" : 2,
                            "crop_size_h" : h_after_crop,
                            "crop_size_w" : w_after_crop,
                            "src_image_h": shape_in_ori[2],
                            "src_image_w":shape_in_ori[3],
                            "resize" : 0, "resize_model" : 0, "resize_output_h" : 32, "resize_output_w" : 32,
                            "padding" : 0, "left_padding_size" : 7, "right_padding_size" : 7, "top_padding_size" : 0, "bottom_padding_size" : 4,
                            "csc_switch" : 1,"rbuv_swap_switch" : 0, "ax_swap_switch" : 0,
                            "matrix_r0c0" : 298, "matrix_r0c1" : 516, "matrix_r0c2" : 0, "matrix_r1c0" : 298, "matrix_r1c1" : -100, "matrix_r1c2" : -208,
                            "matrix_r2c0" : 298, "matrix_r2c1" : 0, "matrix_r2c2" : 409,
                            "input_bias_0" : 16, "input_bias_1" : 128, "input_bias_2" : 128,
                            "mean_chn_0" : 1, "mean_chn_1" : 1, "mean_chn_2" : 1, "mean_chn_3" : 1,
                            "var_reci_chn_0" : 1.0, "var_reci_chn_1" : 1.0,
                            "var_reci_chn_2" : 1.0, "min_chn_0" : 0,
                            "min_chn_1" : 0, "min_chn_2" : 0, "min_chn_3" : 0
                            }

            aipp_config_dict_json = json.dumps(aipp_config_dict)
            aipp_res = aipp_compute(data, None, output_data, aipp_config_dict_json, kernel_name = "aipp")

            Weight = tvm.placeholder(shape_w, name='params_1', dtype=w_dtype,
                                    attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW", "format":weight_format})
            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='params_2', dtype=bias["dtype"])
                fusion_type = 772 if aipp_input_format == "YUV420SP_U8" else 4
            else:
                bias_tensor = None
                fusion_type = 771 if aipp_input_format == "YUV420SP_U8" else 3

            conv_res = conv2d_compute(aipp_res, Weight, bias_tensor, None, None, strides, pads, dilations)
            out = leaky_relu_compute(conv_res, None, negative_slope=0)
            auto_sch_res = AutoScheduleOp(out)
            assert auto_sch_res.fusion_type == fusion_type
            sch = generic.auto_schedule(out)

        kernel_name = "acr_fuse_1910"
        if bias != None:
            tensor_list = [data, Weight, bias_tensor, out]
        else:
            tensor_list = [data, Weight, out]

        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)

    def aipp_conv_relu_maxpooling_template( shape_in_ori, shape_w_ori, \
        in_dtype, w_dtype, strides, pads, dilations, pooling_window, \
        pooling_stride, pooling_padding, bias=None, C04=True,aipp_format="yuv", \
        crops=False, soc_version="Ascend310"):
        N, C, H, W = shape_in_ori

        if soc_version in ["Ascend610", "Ascend710"] and C04:
            shape_in = (N, (C + 3)//4, H, W, 4)
        else:
            shape_in = (N, (C + 15)//16, H, W, 16)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16

        if C04:
            Ci1 = 1
            Ci0 = 4
            hwc1_4 = int_ceil_div(Hk*Wk*Ci1, 4)
            shape_w = (hwc1_4, Co1, Co0, Ci0*4)
            weight_format = "FRACTAL_Z_C04"
        else:
            Ci1 = 1
            Ci0 = 16
            shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)
            weight_format = "FRACTAL_Z"

        if aipp_format == "yuv":
            aipp_input_format = "YUV420SP_U8"
        else:
            aipp_input_format = "RGB888_U8"


        with tvm.target.cce():

            data = tvm.placeholder(shape_in_ori, name="params_0", dtype="uint8",attrs={"ori_shape": shape_in_ori, "format" : "NCHW", "ori_format": "NHWC"})

            if crops:
                h_after_crop = shape_in_ori[2] - 4
                w_after_crop = shape_in_ori[3] - 4
            else:
                h_after_crop = shape_in_ori[2]
                w_after_crop = shape_in_ori[3]
            if soc_version in ["Ascend610", "Ascend710"] and C04:

                output_data = {'shape': [N,1,h_after_crop,w_after_crop,4], 'ori_shape': [N,3,h_after_crop,w_after_crop], 'format': 'NC1HWC0_C04', 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'use_L1_workspace': 0, 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': (), 'split_index': 0}
            else:
                output_data = {'shape': [N,1,h_after_crop,w_after_crop,16], 'ori_shape': [N,3,h_after_crop,w_after_crop], 'format': 'NC1HWC0', 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'use_L1_workspace': 0, 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': (), 'split_index': 0}

            aipp_config_dict = {"cce_product" : "2.1",
                            "out_dtype" : "float16",
                            "out_format" : "NC1HWC0",
                            "aipp_mode" : "static",
                            "related_input_rank" : 0,
                            "input_format" :aipp_input_format,
                            "src_image_size_n" : shape_in_ori[0],
                            "src_image_size_c" : shape_in_ori[1],
                            "src_image_size_h" : shape_in_ori[2],
                            "src_image_size_w" : shape_in_ori[3],
                            "crop" : crops,
                            "load_start_pos_h" : 2,
                            "load_start_pos_w" : 2,
                            "crop_size_h" : h_after_crop,
                            "crop_size_w" : w_after_crop,
                            "src_image_h": shape_in_ori[2],
                            "src_image_w":shape_in_ori[3],
                            "resize" : 0, "resize_model" : 0, "resize_output_h" : 32, "resize_output_w" : 32,
                            "padding" : 0, "left_padding_size" : 7, "right_padding_size" : 7, "top_padding_size" : 0, "bottom_padding_size" : 4,
                            "csc_switch" : 1,"rbuv_swap_switch" : 0, "ax_swap_switch" : 0,
                            "matrix_r0c0" : 298, "matrix_r0c1" : 516, "matrix_r0c2" : 0, "matrix_r1c0" : 298, "matrix_r1c1" : -100, "matrix_r1c2" : -208,
                            "matrix_r2c0" : 298, "matrix_r2c1" : 0, "matrix_r2c2" : 409,
                            "input_bias_0" : 16, "input_bias_1" : 128, "input_bias_2" : 128,
                            "mean_chn_0" : 1, "mean_chn_1" : 1, "mean_chn_2" : 1, "mean_chn_3" : 1,
                            "var_reci_chn_0" : 1.0, "var_reci_chn_1" : 1.0,
                            "var_reci_chn_2" : 1.0, "min_chn_0" : 0,
                            "min_chn_1" : 0, "min_chn_2" : 0, "min_chn_3" : 0
                            }

            aipp_config_dict_json = json.dumps(aipp_config_dict)
            aipp_res = aipp_compute(data, None, output_data, aipp_config_dict_json, kernel_name = "aipp")

            Weight = tvm.placeholder(shape_w, name='params_1', dtype=w_dtype,
                                    attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW", "format":weight_format}) # [13,4,16,16]
            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='params_2', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(aipp_res, Weight, bias_tensor, None, None, strides, pads, dilations)
            relu = leaky_relu_compute(conv_res, None, negative_slope=0)
            out = pool_fuse_compute(relu, None, None, None,pooling_window, pooling_stride, pad=pooling_padding)
            auto_sch_res = AutoScheduleOp(out)
            fusion_type = 813 if aipp_input_format == "YUV420SP_U8" else 45
            assert auto_sch_res.fusion_type == fusion_type
            sch = generic.auto_schedule(out)

        kernel_name = "acr_fuse_1910"
        if bias != None:
            tensor_list = [data, Weight, bias_tensor, out]
        else:
            tensor_list = [data, Weight, out]

        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)

    def aipp_conv_relu_quant_template( shape_in_ori, shape_w_ori, \
        in_dtype, w_dtype, strides, pads, dilations,quant_dict, bias=None, \
        C04=True,aipp_format="yuv", crops=False, soc_version="Ascend310"):
        N, C, H, W = shape_in_ori

        if soc_version in ["Ascend610", "Ascend710"] and C04:
            shape_in = (N, (C + 3)//4, H, W, 4)
        else:
            shape_in = (N, (C + 15)//16, H, W, 16)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16

        if C04:
            Ci1 = 1
            Ci0 = 4
            hwc1_4 = int_ceil_div(Hk*Wk*Ci1, 4)
            shape_w = (hwc1_4, Co1, Co0, Ci0*4)
            weight_format = "FRACTAL_Z_C04"
        else:
            Ci1 = 1
            Ci0 = 16
            shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)
            weight_format = "FRACTAL_Z"

        if aipp_format == "yuv":
            aipp_input_format = "YUV420SP_U8"
        else:
            aipp_input_format = "RGB888_U8"


        with tvm.target.cce():

            data = tvm.placeholder(shape_in_ori, name="params_0", dtype="uint8",attrs={"ori_shape": shape_in_ori, "format" : "NCHW", "ori_format": "NHWC"})

            if crops:
                h_after_crop = shape_in_ori[2] - 4
                w_after_crop = shape_in_ori[3] - 4
            else:
                h_after_crop = shape_in_ori[2]
                w_after_crop = shape_in_ori[3]
            if soc_version in ["Ascend610", "Ascend710"] and C04:

                output_data = {'shape': [N,1,h_after_crop,w_after_crop,4], 'ori_shape': [N,3,h_after_crop,w_after_crop], 'format': 'NC1HWC0_C04', 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'use_L1_workspace': 0, 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': (), 'split_index': 0}
            else:
                output_data = {'shape': [N,1,h_after_crop,w_after_crop,16], 'ori_shape': [N,3,h_after_crop,w_after_crop], 'format': 'NC1HWC0', 'ori_format': 'NCHW', 'dtype': 'float16', 'addr_type': 0, 'valid_shape': (), 'slice_offset': (), 'use_L1_workspace': 0, 'L1_workspace_size': -1, 'L1_fusion_type': -1, 'L1_addr_offset': 0, 'total_shape': (), 'split_index': 0}

            aipp_config_dict = {"cce_product" : "2.1",
                            "out_dtype" : "float16",
                            "out_format" : "NC1HWC0",
                            "aipp_mode" : "static",
                            "related_input_rank" : 0,
                            "input_format" :aipp_input_format,
                            "src_image_size_n" : shape_in_ori[0],
                            "src_image_size_c" : shape_in_ori[1],
                            "src_image_size_h" : shape_in_ori[2],
                            "src_image_size_w" : shape_in_ori[3],
                            "crop" : crops,
                            "load_start_pos_h" : 2,
                            "load_start_pos_w" : 2,
                            "crop_size_h" : h_after_crop,
                            "crop_size_w" : w_after_crop,
                            "src_image_h": shape_in_ori[2],
                            "src_image_w":shape_in_ori[3],
                            "resize" : 0, "resize_model" : 0, "resize_output_h" : 32, "resize_output_w" : 32,
                            "padding" : 0, "left_padding_size" : 7, "right_padding_size" : 7, "top_padding_size" : 0, "bottom_padding_size" : 4,
                            "csc_switch" : 1,"rbuv_swap_switch" : 0, "ax_swap_switch" : 0,
                            "matrix_r0c0" : 298, "matrix_r0c1" : 516, "matrix_r0c2" : 0, "matrix_r1c0" : 298, "matrix_r1c1" : -100, "matrix_r1c2" : -208,
                            "matrix_r2c0" : 298, "matrix_r2c1" : 0, "matrix_r2c2" : 409,
                            "input_bias_0" : 16, "input_bias_1" : 128, "input_bias_2" : 128,
                            "mean_chn_0" : 1, "mean_chn_1" : 1, "mean_chn_2" : 1, "mean_chn_3" : 1,
                            "var_reci_chn_0" : 1.0, "var_reci_chn_1" : 1.0,
                            "var_reci_chn_2" : 1.0, "min_chn_0" : 0,
                            "min_chn_1" : 0, "min_chn_2" : 0, "min_chn_3" : 0
                            }

            aipp_config_dict_json = json.dumps(aipp_config_dict)
            aipp_res = aipp_compute(data, None, output_data, aipp_config_dict_json, kernel_name = "aipp")

            Weight = tvm.placeholder(shape_w, name='params_1', dtype=w_dtype,
                                    attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW", "format":weight_format})
            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='params_2', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(aipp_res, Weight, bias_tensor, None, None, strides, pads, dilations)
            relu = leaky_relu_compute(conv_res, None, negative_slope=0.1)
            out = ascend_quant_compute(relu,
                                        None,
                                        quant_dict['scale'],
                                        quant_dict['sqrt_mode'],
                                        quant_dict['offset'],
                                        quant_dict['round_mode'])
            auto_sch_res = AutoScheduleOp(out)
            if soc_version in ("Ascend610", "Ascend710"):
                fusion_type = 808 if aipp_input_format == "YUV420SP_U8" else 40
            else:
                fusion_type = 809 if aipp_input_format == "YUV420SP_U8" else 41
            assert auto_sch_res.fusion_type == fusion_type
            sch = generic.auto_schedule(out)

        kernel_name = "acr_fuse_1910"
        if bias != None:
            tensor_list = [data, Weight, bias_tensor, out]
        else:
            tensor_list = [data, Weight, out]

        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)

    def aipp_conv_relu_test_cases( soc):
        shape_in_ori = (1, 3, 24, 24) # NC1HWC0
        shape_w_ori = (16, 3, 3, 3) #  HkWkCi1, Co1, 16, 16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [1, 1, 1, 1]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        aipp_conv_relu_template( shape_in_ori, \
            shape_w_ori, in_dtype, w_dtype, \
            strides, pads, dilations, bias=bias, C04=True,aipp_format="yuv", \
            crops=False, soc_version=soc)

        shape_in_ori = (1, 3, 1024, 1824) #  NC1HWC0
        shape_w_ori = (32, 3, 3, 3) #  HkWkCi1, Co1, 16, 16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 2, 2]
        pads = [1, 1, 1, 1]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        aipp_conv_relu_template( shape_in_ori, \
            shape_w_ori, in_dtype, w_dtype, \
            strides, pads, dilations, bias=None, C04=False, aipp_format="yuv", \
            crops=True, soc_version=soc)

        shape_in_ori = (16, 3, 228, 240) #  NC1HWC0
        shape_w_ori = (64, 3, 3, 3) #  HkWkCi1, Co1, 16, 16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 2, 2]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        aipp_conv_relu_template( shape_in_ori, \
            shape_w_ori, in_dtype, w_dtype, \
            strides, pads, dilations, bias=None, C04=False, aipp_format="rgb", \
            crops=True, soc_version=soc)

        # test XRGB
        shape_in_ori = (1, 4, 24, 24) # NC1HWC0
        shape_w_ori = (16, 4, 3, 3) #  HkWkCi1, Co1, 16, 16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [1, 1, 1, 1]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        aipp_conv_relu_template( shape_in_ori, \
            shape_w_ori, in_dtype, w_dtype, \
            strides, pads, dilations, bias=bias, C04=True,aipp_format="xrgb", \
            crops=False, soc_version=soc)

    def aipp_conv_relu_maxpooling_test_cases(soc):

        shape_in_ori = (8, 3, 224, 224) #  NC1HWC0
        shape_w_ori = (64, 3, 7, 7) #  HkWkCi1, Co1, 16, 16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 2, 2]
        pads = [3, 3, 3, 3]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        pooling_window = (3,3)
        pooling_stride = (2,2)
        pooling_padding = [0,1,0,1]
        aipp_conv_relu_maxpooling_template( shape_in_ori, shape_w_ori, \
            in_dtype, w_dtype, strides, pads, \
            dilations, pooling_window, pooling_stride, \
            pooling_padding, bias=bias, C04=True, aipp_format="yuv", \
            crops=False, soc_version=soc)

    def aipp_conv_relu_quant_test_cases(soc):

        shape_in_ori = (1, 3, 416, 416) #  NC1HWC0
        shape_w_ori = (32, 3, 3, 3) #  HkWkCi1, Co1, 16, 16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [1, 1, 1, 1]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        quant_dict = {
        'scale': 10.32,
        'sqrt_mode': True,
        'offset': 0.5,
        'round_mode':'Round'
        }
        aipp_conv_relu_quant_template( shape_in_ori, shape_w_ori, in_dtype, \
            w_dtype, strides, pads, dilations,quant_dict, \
            bias=bias, C04=True,aipp_format="yuv", crops=False, soc_version=soc)

    def aipp_conv_relu_test_multiplatforms():
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            aipp_conv_relu_test_cases(soc)

    def aipp_conv_relu_quant_test_multiplatforms():
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            aipp_conv_relu_quant_test_cases(soc)

    def aipp_conv_relu_maxpooling_test_multiplatforms():
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            aipp_conv_relu_maxpooling_test_cases(soc)

    def aipp_conv_leakyrelu_l1fusion_test():
        json_str = '{\"SocInfo\":{\"autoTilingMode\":\"\",\"coreNum\":\"\",\"coreType\":\"AiCore\",\"deviceId\":\"\",\"l1Fusion\":\"true\",\"l2Fusion\":\"false\",\"l2Mode\":\"0\",\"op_debug_level\":\"1\",\"op_impl_mode\":\"high_performance\",\"op_impl_mode_list\":[],\"socVersion\":\"SD3403\",\"vector_fp_ceiling\":\"2\"},\"fusion_op_name\":\"te_fused_op_aipp_conv2d_leaky_relu_ascend_quant_d16492ba463ee5c4_5a5710b828510564_0\",\"graph_name\":\"partition0_rank1_new_sub_graph1\",\"l1_size\":262144,\"op_list\":[{\"name\":\"-1_0_new_sub_graph1_PlaceHolder0__0\",\"output_desc\":[{\"L1_addr_flag\":1,\"L1_addr_offset\":262144,\"L1_fusion_type\":0,\"L1_valid_size\":98304,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"uint8\",\"format\":\"NHWC\",\"name\":\"-1_0_new_sub_graph1_PlaceHolder0__0\",\"ori_format\":\"NHWC\",\"ori_shape\":[1,96,128,3],\"shape\":[1,96,128,3],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,96,128,3],\"valid_shape\":[]}],\"type\":\"Data\"},{\"name\":\"-1_0_data_0_aipp_lxfuse0OPT\",\"output_desc\":[{\"data_type\":0,\"name\":\"-1_0_data_0_aipp_lxfuse0OPT\",\"shape\":\"NULL\"}],\"type\":\"Data\"},{\"name\":\"-1_0_new_sub_graph1_PlaceHolder1__0\",\"output_desc\":[{\"L1_addr_flag\":0,\"L1_addr_offset\":-1,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"FRACTAL_Z_C04\",\"name\":\"-1_0_new_sub_graph1_PlaceHolder1__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[32,3,5,5],\"shape\":[7,2,16,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[7,2,16,16],\"valid_shape\":[]}],\"type\":\"Data\"},{\"name\":\"-1_0_new_sub_graph1_PlaceHolder2__0\",\"output_desc\":[{\"L1_addr_flag\":0,\"L1_addr_offset\":-1,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NCHW\",\"name\":\"-1_0_new_sub_graph1_PlaceHolder2__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[32],\"shape\":[32],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[32],\"valid_shape\":[]}],\"type\":\"Data\"},{\"name\":\"-1_0_l1-c_lxfuse0OPT\",\"output_desc\":[{\"data_type\":0,\"name\":\"-1_0_l1-c_lxfuse0OPT\",\"shape\":\"NULL\"}],\"type\":\"Data\"},{\"attr_desc\":[\"{\\"aipp_mode\\":\\"static\\",\\"input_format\\":\\"RGB888_U8\\",\\"src_image_size_h\\":96,\\"src_image_size_w\\":128}\"],\"dynamic_compile_static\":false,\"func_name\":\"aipp\",\"id\":1,\"input_desc\":[{\"L1_addr_flag\":1,\"L1_addr_offset\":262144,\"L1_fusion_type\":0,\"L1_valid_size\":98304,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"uint8\",\"format\":\"NHWC\",\"name\":\"-1_0_new_sub_graph1_PlaceHolder0__0\",\"ori_format\":\"NHWC\",\"ori_shape\":[1,96,128,3],\"shape\":[1,96,128,3],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,96,128,3],\"valid_shape\":[]},{\"data_type\":0,\"name\":\"-1_0_data_0_aipp_lxfuse0OPT\",\"shape\":\"NULL\"}],\"int64mode\":false,\"module_name\":\"impl.aipp\",\"name\":\"-1_0_data_0_aipp_lxfuse0\",\"ori_name\":[\"data_0_aipp\"],\"output_data_desc\":[{\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"dtype\":\"float16\",\"format\":\"NC1HWC0_C04\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,3,96,128],\"shape\":[1,1,96,128,4],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,1,96,128,4],\"valid_shape\":[]}],\"output_desc\":[{\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NC1HWC0_C04\",\"name\":\"-1_0_data_0_aipp_lxfuse0__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,3,96,128],\"output_index\":0,\"shape\":[1,1,96,128,4],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,1,96,128,4],\"valid_shape\":[]}],\"pattern\":\"aipp\",\"py_module_path\":\"\",\"type\":\"Aipp\"},{\"attr_desc\":[[1,1,2,2],[2,2,2,2],[1,1,1,1],1,\"NCHW\",0],\"dynamic_compile_static\":false,\"func_name\":\"conv2d\",\"id\":4,\"input_desc\":[{\"L1_addr_flag\":0,\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NC1HWC0_C04\",\"name\":\"-1_0_data_0_aipp_lxfuse0__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,3,96,128],\"peer_out_param_index\":1,\"shape\":[1,1,96,128,4],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,1,96,128,4],\"valid_shape\":[]},{\"L1_addr_flag\":0,\"L1_addr_offset\":-1,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"FRACTAL_Z_C04\",\"name\":\"-1_0_new_sub_graph1_PlaceHolder1__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[32,3,5,5],\"peer_out_param_index\":2,\"shape\":[7,2,16,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[7,2,16,16],\"valid_shape\":[]},{\"L1_addr_flag\":0,\"L1_addr_offset\":-1,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NCHW\",\"name\":\"-1_0_new_sub_graph1_PlaceHolder2__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[32],\"peer_out_param_index\":3,\"shape\":[32],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[32],\"valid_shape\":[]},{\"data_type\":0,\"name\":\"-1_0_l1-c_lxfuse0OPT\",\"shape\":\"NULL\"}],\"int64mode\":false,\"module_name\":\"impl.conv2d\",\"name\":\"-1_0_l1-c_lxfuse0\",\"ori_name\":[\"l1-c_dequant_layer\",\"l1-c\",\"l1-c_0_quant_layer\"],\"output_data_desc\":[{\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"dtype\":\"float16\",\"format\":\"NC1HWC0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"shape\":[1,2,48,64,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,2,48,64,16],\"valid_shape\":[]}],\"output_desc\":[{\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NC1HWC0\",\"name\":\"-1_0_l1-c_lxfuse0__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"output_index\":0,\"shape\":[1,2,48,64,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,2,48,64,16],\"valid_shape\":[]}],\"pattern\":\"Convolution\",\"py_module_path\":\"\",\"type\":\"Conv2D\"},{\"attr_desc\":[0.0],\"dynamic_compile_static\":false,\"func_name\":\"leaky_relu\",\"id\":5,\"input_desc\":[{\"L1_addr_flag\":0,\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NC1HWC0\",\"name\":\"-1_0_l1-c_lxfuse0__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"shape\":[1,2,48,64,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,2,48,64,16],\"valid_shape\":[]}],\"int64mode\":false,\"module_name\":\"impl.leaky_relu\",\"name\":\"-1_0_l1-a_lxfuse0\",\"ori_name\":[\"l1-a\"],\"output_data_desc\":[{\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"dtype\":\"float16\",\"format\":\"NC1HWC0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"shape\":[1,2,48,64,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,2,48,64,16],\"valid_shape\":[]}],\"output_desc\":[{\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NC1HWC0\",\"name\":\"-1_0_l1-a_lxfuse0__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"output_index\":0,\"shape\":[1,2,48,64,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,2,48,64,16],\"valid_shape\":[]}],\"pattern\":\"ElemWise\",\"py_module_path\":\"\",\"type\":\"LeakyRelu\"},{\"attr_desc\":[10.281484603881836,-128.0,false,\"Round\"],\"dynamic_compile_static\":false,\"func_name\":\"ascend_quant\",\"id\":6,\"input_desc\":[{\"L1_addr_flag\":0,\"L1_addr_offset\":0,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":0,\"data_type\":\"float16\",\"format\":\"NC1HWC0\",\"name\":\"-1_0_l1-a_lxfuse0__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"shape\":[1,2,48,64,16],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,2,48,64,16],\"valid_shape\":[]}],\"int64mode\":false,\"module_name\":\"impl.ascend_quant\",\"name\":\"-1_0_l2-c_0_quant_layer_lxfuse0\",\"ori_name\":[\"l2-c_0_quant_layer\"],\"output_data_desc\":[{\"L1_addr_offset\":950272,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":1,\"dtype\":\"int8\",\"format\":\"NC1HWC0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"shape\":[1,1,48,64,32],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,1,48,64,32],\"valid_shape\":[]}],\"output_desc\":[{\"L1_addr_offset\":950272,\"L1_fusion_type\":0,\"L1_workspace_size\":-1,\"addr_type\":1,\"data_type\":\"int8\",\"format\":\"NC1HWC0\",\"name\":\"-1_0_l2-c_0_quant_layer_lxfuse0__0\",\"ori_format\":\"NCHW\",\"ori_shape\":[1,32,48,64],\"output_index\":0,\"shape\":[1,1,48,64,32],\"slice_offset\":[],\"split_index\":0,\"sub_format\":0,\"total_shape\":[1,1,48,64,32],\"valid_shape\":[]}],\"pattern\":\"quant\",\"py_module_path\":\"\",\"type\":\"AscendQuant\"}],\"scope_id\":1}'
        
        op_desc = json.loads(json_str)
        soc = op_desc['SocInfo']
        cce_conf.te_set_version(soc['socVersion'],
                        soc['coreType'],
                        soc['coreNum'],
                        soc['l1Fusion'],
                        soc['l2Mode'],
                        soc['l2Fusion'])
                        
        fusion_op(json_str)


    print("[ aipp_conv_relu ]")
    aipp_conv_relu_test_multiplatforms()

    print("[ aipp_conv_relu_quant ]")
    aipp_conv_relu_quant_test_multiplatforms()

    print("[ aipp_conv_leakyrelu l1fusion ]")
    aipp_conv_leakyrelu_l1fusion_test()


print("adding Conv2D aipp testcases")
ut_case.add_cust_test_func(test_func=test_conv_aipp)
