#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def TestConvSreadSwriteConf(test_arg):
    """
    Copyright 2018 Huawei Technologies Co., Ltd
    strided_read + conv + strided_write ut
    """
    import impl
    import unittest
    import json
    import te
    from te import tvm
    from tbe.common import utils
    import te.lang.cce
    from te.lang.cce import cce_build_code
    from tbe.dsl import auto_schedule
    from te.platform import cce_conf
    from te.platform.fusion_util import fusion_op
    from te.platform.fusion_manager import fusion_manager
    from impl.strided_write import strided_write_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.ascend_requant import ascend_requant_compute
    from impl.conv2d import conv2d_compute
    from impl.leaky_relu import leaky_relu_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp


    def strided_read_compute(input_tensor, output_tensor, axis, stride, \
        kernel_name='strided_read'):
        output_y = tvm.compute(
            output_tensor.get("shape"),
            lambda batch_idx, c1_idx, h_idx, w_idx, c0_idx:
            input_tensor[batch_idx, c1_idx, h_idx, w_idx, c0_idx],
            name="strided_read",
            tag="strided_read",
            attrs={"ori_shape": [], "ori_format": "NCHW"})

        return output_y


    # conv + stridedwrite
    def conv_bias_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, \
            attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder((shape_w[1]*shape_w[2], ), \
                    name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)
            output_tensor = {"shape": tuple(i.value for i in conv_res.shape)}
            conv_swrite_res = strided_write_compute(conv_res, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)
        kernel_name = "conv2d_swrite"
        if bias is not None:
            tensor_list = [data, weight, bias_tensor, conv_swrite_res]
        else:
            tensor_list = [data, weight, conv_swrite_res]
        if bias is not None:
            fusion_type = 2
        else:
            fusion_type = 1
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    def conv_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = None
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, \
            attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder((shape_w[1]*shape_w[2], ), \
                    name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)
            output_tensor = {"shape": tuple(i.value for i in conv_res.shape)}
            conv_swrite_res = strided_write_compute(conv_res, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)
        kernel_name = "conv2d_swrite"
        if bias is not None:
            tensor_list = [data, weight, bias_tensor, conv_swrite_res]
        else:
            tensor_list = [data, weight, conv_swrite_res]
        if bias is not None:
            fusion_type = 2
        else:
            fusion_type = 1
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    # stridedread+conv+stridedwrite
    def sread_conv_bias_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        stride_sread = 128
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data_ori = tvm.placeholder((batch_in, stride_sread, h_in, w_in, ci0), \
            name='Fmap', dtype=in_dtype) # A_DDR
            data = strided_read_compute(data_ori, {"shape": shape_in}, 1, \
            stride_sread, "strided_read") # AL1
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)
            output_tensor = {"shape": tuple(i.value for i in conv_res.shape)}
            conv_swrite_res = strided_write_compute(conv_res, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)

        kernel_name = "sread_conv_swrite"
        if bias is not None:
            tensor_list = [data_ori, weight, bias_tensor, \
            sch.cce_special["real_out_tensor"][0]]
            fusion_type = 2
        else:
            tensor_list = [data_ori, weight, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 1
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    def sread_conv_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = None
        stride_sread = 128
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data_ori = tvm.placeholder((batch_in, stride_sread, h_in, w_in, ci0), \
            name='Fmap', dtype=in_dtype) # A_DDR
            data = strided_read_compute(data_ori, {"shape": shape_in}, 1, \
            stride_sread, "strided_read") # AL1
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)
            output_tensor = {"shape": tuple(i.value for i in conv_res.shape)}
            conv_swrite_res = strided_write_compute(conv_res, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)

        kernel_name = "sread_conv_swrite"
        if bias is not None:
            tensor_list = [data_ori, weight, bias_tensor, \
            sch.cce_special["real_out_tensor"][0]]
            fusion_type = 2
        else:
            tensor_list = [data_ori, weight, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 1
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    #===========================relu=========================
    def conv_bias_relu_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, \
            attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder((shape_w[1]*shape_w[2], ), \
                    name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)

            relu = leaky_relu_compute(conv_res, None, 0.0)

            output_tensor = {"shape": tuple(i.value for i in relu.shape)}

            conv_swrite_res = strided_write_compute(relu, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)

        if bias is not None:
            tensor_list = [data, weight, bias_tensor, conv_swrite_res]
            fusion_type = 4
        else:
            tensor_list = [data, weight, conv_swrite_res]
            fusion_type = 3
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d_relu_swrite",
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    def conv_relu_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = None
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, \
            attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder((shape_w[1]*shape_w[2], ), \
                    name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)

            relu = leaky_relu_compute(conv_res, None, 0.0)

            output_tensor = {"shape": tuple(i.value for i in relu.shape)}

            conv_swrite_res = strided_write_compute(relu, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)

        if bias is not None:
            tensor_list = [data, weight, bias_tensor, conv_swrite_res]
            fusion_type = 4
        else:
            tensor_list = [data, weight, conv_swrite_res]
            fusion_type = 3
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d_relu_swrite",
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    def sread_conv_bias_relu_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        stride_sread = 128
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data_ori = tvm.placeholder((batch_in, stride_sread, h_in, w_in, ci0), \
            name='Fmap', dtype=in_dtype) # A_DDR
            data = strided_read_compute(data_ori, {"shape": shape_in}, 1, \
                stride_sread, "strided_read")
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)
            relu = leaky_relu_compute(conv_res, None, 0.0)
            output_tensor = {"shape": tuple(i.value for i in relu.shape)}
            conv_swrite_res = strided_write_compute(relu, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)


        if bias is not None:
            tensor_list = [data_ori, weight, bias_tensor, \
            sch.cce_special["real_out_tensor"][0]]
            fusion_type = 4
        else:
            tensor_list = [data_ori, weight, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 3
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": "sread_conv_relu_swrite",
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def sread_conv_relu_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = None
        stride_sread = 128
        stride_swrite = 128

        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data_ori = tvm.placeholder((batch_in, stride_sread, h_in, w_in, ci0), \
            name='Fmap', dtype=in_dtype) # A_DDR
            data = strided_read_compute(data_ori, {"shape": shape_in}, 1, \
                stride_sread, "strided_read")
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)
            relu = leaky_relu_compute(conv_res, None, 0.0)
            output_tensor = {"shape": tuple(i.value for i in relu.shape)}
            conv_swrite_res = strided_write_compute(relu, output_tensor, 1, \
                stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)


        if bias is not None:
            tensor_list = [data_ori, weight, bias_tensor, \
            sch.cce_special["real_out_tensor"][0]]
            fusion_type = 4
        else:
            tensor_list = [data_ori, weight, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 3
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": "sread_conv_relu_swrite",
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    #========================dequant==========================

    # stridedread+conv+dequant+stridedwrite
    def sread_conv_bias_dequant_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        dequant_dict = {
        'deq_scale_shape': [1, 32, 1, 1, 16], # 1, co1, 1, 1, 16
        'deq_scale_dtype': "uint64",
        'sqrt_mode': False,
        'relu_flag': False,
        }
        quant_dict = {
        'scale': 1,
        'sqrt_mode': False,
        'offset': 0.5,
        'round_mode': 'Round'
        }
        bias = {"dtype": "int32"}
        stride_sread = 128
        stride_swrite = 128
        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data_ori = tvm.placeholder((batch_in, stride_sread, h_in, w_in, ci0), \
            name='Fmap', dtype=in_dtype) # A_DDR
            data = strided_read_compute(data_ori, {"shape": shape_in}, 1, \
                stride_sread, "strided_read")
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})

            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)

            dequant_scale = tvm.placeholder(dequant_dict['deq_scale_shape'], \
                dequant_dict['deq_scale_dtype'], "deq_scale", attrs={'ori_shape':[c_out]})

            dequant_res = ascend_dequant_compute(conv_res,
                                                dequant_scale,
                                                None,
                                                dequant_dict['sqrt_mode'],
                                                dequant_dict['relu_flag'])

            output_tensor = {"shape": tuple(i.value for i in dequant_res.shape)}
            conv_swrite_res = strided_write_compute(dequant_res, output_tensor, \
                1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)

        kernel_name = "sread_conv_dequant_quant_swrite"

        if bias is not None:
            tensor_list = [data_ori, weight, bias_tensor, dequant_scale, \
            conv_swrite_res]
            fusion_type = 10
        else:
            tensor_list = [data_ori, weight, dequant_scale, conv_swrite_res]
            fusion_type = 9
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def sread_conv_dequant_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        dequant_dict = {
        'deq_scale_shape': [1, 32, 1, 1, 16], # 1, co1, 1, 1, 16
        'deq_scale_dtype': "uint64",
        'sqrt_mode': False,
        'relu_flag': False,
        }
        quant_dict = {
        'scale': 1,
        'sqrt_mode': False,
        'offset': 0.5,
        'round_mode': 'Round'
        }
        bias = None
        stride_sread = 128
        stride_swrite = 128
        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data_ori = tvm.placeholder((batch_in, stride_sread, h_in, w_in, ci0), \
            name='Fmap', dtype=in_dtype) # A_DDR
            data = strided_read_compute(data_ori, {"shape": shape_in}, 1, \
                stride_sread, "strided_read")
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})

            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)

            dequant_scale = tvm.placeholder(dequant_dict['deq_scale_shape'], \
                dequant_dict['deq_scale_dtype'], "deq_scale", attrs={'ori_shape':[c_out]})

            dequant_res = ascend_dequant_compute(conv_res,
                                                dequant_scale,
                                                None,
                                                dequant_dict['sqrt_mode'],
                                                dequant_dict['relu_flag'])

            output_tensor = {"shape": tuple(i.value for i in dequant_res.shape)}
            conv_swrite_res = strided_write_compute(dequant_res, output_tensor, \
                1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)

        kernel_name = "sread_conv_dequant_quant_swrite"

        if bias is not None:
            tensor_list = [data_ori, weight, bias_tensor, dequant_scale, \
            conv_swrite_res]
            fusion_type = 10
        else:
            tensor_list = [data_ori, weight, dequant_scale, conv_swrite_res]
            fusion_type = 9
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    # conv+dequant+stridedwrite
    def conv_bias_dequant_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        dequant_dict = {
        'deq_scale_shape': [1, 32, 1, 1, 16], # 1, co1, 1, 1, 16
        'deq_scale_dtype': "uint64",
        'sqrt_mode': False,
        'relu_flag': False,
        }
        quant_dict = {
        'scale': 1,
        'sqrt_mode': False,
        'offset': 0.5,
        'round_mode': 'Round'
        }
        bias = {"dtype": "int32"}
        stride_sread = 128
        stride_swrite = 128
        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, \
            attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)

            dequant_scale = tvm.placeholder(dequant_dict['deq_scale_shape'], \
                dequant_dict['deq_scale_dtype'], "deq_scale", attrs={'ori_shape':[c_out]})

            dequant_res = ascend_dequant_compute(conv_res,
                                                dequant_scale,
                                                None,
                                                dequant_dict['sqrt_mode'],
                                                dequant_dict['relu_flag'])
            output_tensor = {"shape": tuple(i.value for i in dequant_res.shape)}
            conv_swrite_res = strided_write_compute(dequant_res, output_tensor, \
                1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)
        kernel_name = "conv_dequant_swrite"
        if bias is not None:
            tensor_list = [data, weight, bias_tensor, dequant_scale, \
            conv_swrite_res]
            fusion_type = 10
        else:
            tensor_list = [data, weight, dequant_scale, conv_swrite_res]
            fusion_type = 9
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    def conv_dequant_swrite():
        shape_in_ori = (2, 1024, 14, 14)
        shape_w_ori = (512, 1024, 3, 3)
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        dequant_dict = {
        'deq_scale_shape': [1, 32, 1, 1, 16], # 1, co1, 1, 1, 16
        'deq_scale_dtype': "uint64",
        'sqrt_mode': False,
        'relu_flag': False,
        }
        quant_dict = {
        'scale': 1,
        'sqrt_mode': False,
        'offset': 0.5,
        'round_mode': 'Round'
        }
        bias = None
        stride_sread = 128
        stride_swrite = 128
        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, \
            attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)

            dequant_scale = tvm.placeholder(dequant_dict['deq_scale_shape'], \
                dequant_dict['deq_scale_dtype'], "deq_scale", attrs={'ori_shape':[c_out]})

            dequant_res = ascend_dequant_compute(conv_res,
                                                dequant_scale,
                                                None,
                                                dequant_dict['sqrt_mode'],
                                                dequant_dict['relu_flag'])
            output_tensor = {"shape": tuple(i.value for i in dequant_res.shape)}
            conv_swrite_res = strided_write_compute(dequant_res, output_tensor, \
                1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)
        kernel_name = "conv_dequant_swrite"
        if bias is not None:
            tensor_list = [data, weight, bias_tensor, dequant_scale, \
            conv_swrite_res]
            fusion_type = 10
        else:
            tensor_list = [data, weight, dequant_scale, conv_swrite_res]
            fusion_type = 9
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    # stridedread+strideh_opti
    def stridedread_strideh_opti_int8():
        shape_in_ori = (2, 1024, 36, 44) # ci1 = 32
        shape_w_ori = (2048, 1024, 1, 1) # co1 = 128
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [2, 2, 2, 2]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        dequant_dict = {
        'deq_scale_shape':[1, 128, 1, 1, 16], # 1, co1, 1, 1, 16
        'deq_scale_dtype': "uint64",
        'sqrt_mode': False,
        'relu_flag': False,
        }
        quant_dict = {
        'scale': 1,
        'sqrt_mode': False,
        'offset': 0.5,
        'round_mode':'Round'
        }
        bias = {"dtype": "int32"}
        stride_sread = 32
        stride_swrite = 256
        batch_in, c_in, h_in, w_in = shape_in_ori
        if in_dtype == "float16":
            shape_in = (batch_in, (c_in + 15)//16, h_in, w_in, 16)
        elif in_dtype == "int8":
            shape_in = (batch_in, (c_in + 31)//32, h_in, w_in, 32)

        c_out, c_in, h_k, w_k = shape_w_ori
        co1 = (c_out + 15)//16
        co0 = 16
        if w_dtype == "float16":
            ci1 = (c_in + 15)//16
            ci0 = 16
        elif w_dtype == "int8":
            ci1 = (c_in + 31)//32
            ci0 = 32
        shape_w = (h_k*w_k*ci1, co1, co0, ci0)

        with tvm.target.cce():
            data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, \
                attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"})
            weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, \
                attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})

            if bias is not None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(data, weight, bias_tensor, None, None, \
                strides, pads, dilations)

            dequant_scale = tvm.placeholder(dequant_dict['deq_scale_shape'], \
                dequant_dict['deq_scale_dtype'], "deq_scale", attrs={'ori_shape':[c_out]})

            dequant_res = ascend_dequant_compute(conv_res,
                                                dequant_scale,
                                                None,
                                                dequant_dict['sqrt_mode'],
                                                dequant_dict['relu_flag'])

            quant_res = ascend_quant_compute(dequant_res,
                                        None,
                                        quant_dict['scale'],
                                        quant_dict['sqrt_mode'],
                                        quant_dict['offset'],
                                        quant_dict['round_mode'])

            output_tensor = {"shape": tuple(i.value for i in quant_res.shape)}
            conv_swrite_res = strided_write_compute(quant_res, output_tensor, \
                1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite_res)
            sch = auto_schedule(conv_swrite_res)

        kernel_name = "sread_conv_dequant_quant_swrite"

        if bias is not None:
            tensor_list = [data, weight, bias_tensor, dequant_scale, \
            conv_swrite_res]
            fusion_type = 12
        else:
            tensor_list = [data, weight, dequant_scale, conv_swrite_res]
            fusion_type = 11
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    def stridedread_strideh_opti_fp16():
        shape_in_ori = (2, 128, 16, 16) # cin=8
        shape_w_ori = (256, 128, 1, 1)  # cout=16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 2, 2]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        stride_sread = 32
        stride_swrite = 32
        N, C, H, W = shape_in_ori

        if in_dtype == "float16":
            shape_in = (N, (C + 15)//16, H, W, 16)
        elif in_dtype == "int8":
            shape_in = (N, (C + 31)//32, H, W, 32)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16
        if w_dtype == "float16":
            Ci1 = (Cin + 15)//16
            Ci0 = 16
        elif w_dtype == "int8":
            Ci1 = (Cin + 31)//32
            Ci0 = 32
        shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        with tvm.target.cce():
            Data_ori = tvm.placeholder((N, stride_sread, H, W, Ci0), name='Fmap', dtype=in_dtype) # A_DDR
            Data = strided_read_compute(Data_ori, {"shape": shape_in}, 1, stride_sread, "strided_read") # AL1
            Weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(Data, Weight, bias_tensor, None, None, strides, pads, dilations)
            y = {"shape": tuple(i.value for i in conv_res.shape)}
            conv_swrite = strided_write_compute(conv_res, y, 1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite)
            sch = auto_schedule(conv_swrite)

        kernel_name = "sread_conv_swrite"
        if bias != None:
            tensor_list = [Data_ori, Weight, bias_tensor, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 2
        else:
            tensor_list = [Data_ori, Weight, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 1
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    # stridedread+al1_load2d
    def stridedread_al1_load2d():
        shape_in_ori = (2, 128, 16, 16) # cin=8
        shape_w_ori = (256, 128, 1, 1)  # cout=16
        in_dtype = "float16"
        w_dtype = "float16"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "float16"}
        stride_sread = 32
        stride_swrite = 32

        N, C, H, W = shape_in_ori
        if in_dtype == "float16":
            shape_in = (N, (C + 15)//16, H, W, 16)
        elif in_dtype == "int8":
            shape_in = (N, (C + 31)//32, H, W, 32)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16
        if w_dtype == "float16":
            Ci1 = (Cin + 15)//16
            Ci0 = 16
        elif w_dtype == "int8":
            Ci1 = (Cin + 31)//32
            Ci0 = 32
        shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        with tvm.target.cce():
            Data_ori = tvm.placeholder((N, stride_sread, H, W, Ci0), name='Fmap', dtype=in_dtype) # A_DDR
            Data = strided_read_compute(Data_ori, {"shape": shape_in}, 1, stride_sread, "strided_read") # AL1
            Weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})
            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None
            conv_res = conv2d_compute(Data, Weight, bias_tensor, None, None, strides, pads, dilations)
            y = {"shape": tuple(i.value for i in conv_res.shape)}
            conv_swrite = strided_write_compute(conv_res, y, 1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite)
            sch = auto_schedule(conv_swrite)

        kernel_name = "sread_conv_swrite"
        if bias != None:
            tensor_list = [Data_ori, Weight, bias_tensor, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 2
        else:
            tensor_list = [Data_ori, Weight, sch.cce_special["real_out_tensor"][0]]
            fusion_type = 1
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    # conv+requant+stridedwrite
    def conv_requant_relu_swrite():
        shape_in_ori = (2, 1024, 14, 14) # shape_in = (2, 64, 14, 14, 16)    NC1HWC0
        shape_w_ori = (512, 1024, 3, 3) # shape_w = (64, 32, 16, 16)         HkWkCi1, Co1, 16, 16
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "int32"}
        stride_sread = 128
        stride_swrite = 128
        relu_flag = True

        N, C, H, W = shape_in_ori
        if in_dtype == "float16":
            shape_in = (N, (C + 15)//16, H, W, 16)
        elif in_dtype == "int8":
            shape_in = (N, (C + 31)//32, H, W, 32)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16
        if w_dtype == "float16":
            Ci1 = (Cin + 15)//16
            Ci0 = 16
        elif w_dtype == "int8":
            Ci1 = (Cin + 31)//32
            Ci0 = 32
        shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        # 走dsl接口，A为5HD，B为小n大Z
        with tvm.target.cce():
            Data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            Weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})

            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(Data, Weight, bias_tensor, None, None, strides, pads, dilations)

            shape_req = (1, Co1, 1, 1, 16)

            vreq_reg = tvm.placeholder(shape_req, name='vreq_reg', dtype='uint64', attrs={'ori_shape':[Cout]})

            requant_res = ascend_requant_compute(conv_res, vreq_reg, None, relu_flag=relu_flag)

            y = {"shape": tuple(i.value for i in requant_res.shape)}

            conv_swrite = strided_write_compute(requant_res, y, 1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite)
            sch = auto_schedule(conv_swrite)

        kernel_name = "conv_requant"

        if bias != None:
            tensor_list = [Data, Weight, bias_tensor, vreq_reg, conv_swrite]
            fusion_type = 27
        else:
            tensor_list = [Data, Weight, vreq_reg, conv_swrite]
            fusion_type = 26
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def conv_requant_swrite():
        shape_in_ori = (2, 1024, 14, 14) # shape_in = (2, 64, 14, 14, 16)    NC1HWC0
        shape_w_ori = (512, 1024, 3, 3) # shape_w = (64, 32, 16, 16)         HkWkCi1, Co1, 16, 16
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "int32"}
        stride_sread = 128
        stride_swrite = 128
        relu_flag = False

        N, C, H, W = shape_in_ori
        if in_dtype == "float16":
            shape_in = (N, (C + 15)//16, H, W, 16)
        elif in_dtype == "int8":
            shape_in = (N, (C + 31)//32, H, W, 32)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16
        if w_dtype == "float16":
            Ci1 = (Cin + 15)//16
            Ci0 = 16
        elif w_dtype == "int8":
            Ci1 = (Cin + 31)//32
            Ci0 = 32
        shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        with tvm.target.cce():
            Data = tvm.placeholder(shape_in, name='Fmap', dtype=in_dtype, attrs={"ori_shape": shape_in_ori, "ori_format": "NCHW"}) # A_DDR
            Weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})

            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(Data, Weight, bias_tensor, None, None, strides, pads, dilations)

            shape_req = (1, Co1, 1, 1, 16)

            vreq_reg = tvm.placeholder(shape_req, name='vreq_reg', dtype='uint64', attrs={'ori_shape':[Cout]})

            requant_res = ascend_requant_compute(conv_res, vreq_reg, None, relu_flag=relu_flag)

            y = {"shape": tuple(i.value for i in requant_res.shape)}

            conv_swrite = strided_write_compute(requant_res, y, 1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite)
            sch = auto_schedule(conv_swrite)

        kernel_name = "conv_requant"

        if bias != None:
            tensor_list = [Data, Weight, bias_tensor, vreq_reg, conv_swrite]
            fusion_type = 27
        else:
            tensor_list = [Data, Weight, vreq_reg, conv_swrite]
            fusion_type = 26
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    # strided_read+conv+requant+stridedwrite
    def sread_conv_requant_relu_swrite():
        shape_in_ori = (2, 1024, 14, 14) # shape_in = (2, 64, 14, 14, 16)    NC1HWC0
        shape_w_ori = (512, 1024, 3, 3) # shape_w = (64, 32, 16, 16)         HkWkCi1, Co1, 16, 16
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "int32"}
        stride_sread = 128
        stride_swrite = 128
        relu_flag = True

        N, C, H, W = shape_in_ori
        if in_dtype == "float16":
            shape_in = (N, (C + 15)//16, H, W, 16)
        elif in_dtype == "int8":
            shape_in = (N, (C + 31)//32, H, W, 32)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16
        if w_dtype == "float16":
            Ci1 = (Cin + 15)//16
            Ci0 = 16
        elif w_dtype == "int8":
            Ci1 = (Cin + 31)//32
            Ci0 = 32
        shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        # 走dsl接口，A为5HD，B为小n大Z
        with tvm.target.cce():
            Data_ori = tvm.placeholder((N, stride_sread, H, W, Ci0), name='Fmap', dtype=in_dtype) # A_DDR
            Data = strided_read_compute(Data_ori, {"shape": shape_in}, 1, stride_sread, "strided_read")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})

            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(Data, Weight, bias_tensor, None, None, strides, pads, dilations)

            shape_req = (1, Co1, 1, 1, 16)

            vreq_reg = tvm.placeholder(shape_req, name='vreq_reg', dtype='uint64', attrs={'ori_shape':[Cout]})

            requant_res = ascend_requant_compute(conv_res, vreq_reg, None, relu_flag=relu_flag)

            y = {"shape": tuple(i.value for i in requant_res.shape)}

            conv_swrite = strided_write_compute(requant_res, y, 1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite)
            sch = auto_schedule(conv_swrite)

        kernel_name = "conv_requant"

        if bias != None:
            tensor_list = [Data_ori, Weight, bias_tensor, vreq_reg, conv_swrite]
            fusion_type = 27
        else:
            tensor_list = [Data_ori, Weight, vreq_reg, conv_swrite]
            fusion_type = 26
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def sread_conv_requant_swrite():
        shape_in_ori = (2, 1024, 14, 14) # shape_in = (2, 64, 14, 14, 16)    NC1HWC0
        shape_w_ori = (512, 1024, 3, 3) # shape_w = (64, 32, 16, 16)         HkWkCi1, Co1, 16, 16
        in_dtype = "int8"
        w_dtype = "int8"
        strides = [1, 1, 1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1, 1, 1]
        bias = {"dtype": "int32"}
        stride_sread = 128
        stride_swrite = 128
        relu_flag = False

        N, C, H, W = shape_in_ori
        if in_dtype == "float16":
            shape_in = (N, (C + 15)//16, H, W, 16)
        elif in_dtype == "int8":
            shape_in = (N, (C + 31)//32, H, W, 32)

        Cout, Cin, Hk, Wk = shape_w_ori
        Co1 = (Cout + 15)//16
        Co0 = 16
        if w_dtype == "float16":
            Ci1 = (Cin + 15)//16
            Ci0 = 16
        elif w_dtype == "int8":
            Ci1 = (Cin + 31)//32
            Ci0 = 32
        shape_w = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        # 走dsl接口，A为5HD，B为小n大Z
        with tvm.target.cce():
            Data_ori = tvm.placeholder((N, stride_sread, H, W, Ci0), name='Fmap', dtype=in_dtype) # A_DDR
            Data = strided_read_compute(Data_ori, {"shape": shape_in}, 1, stride_sread, "strided_read")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype=w_dtype, attrs={"ori_shape": shape_w_ori, "ori_format": "NCHW"})

            if bias != None:
                bias_tensor = tvm.placeholder(
                (shape_w[1]*shape_w[2], ), name='bias_tensor', dtype=bias["dtype"])
            else:
                bias_tensor = None

            conv_res = conv2d_compute(Data, Weight, bias_tensor, None, None, strides, pads, dilations)

            shape_req = (1, Co1, 1, 1, 16)

            vreq_reg = tvm.placeholder(shape_req, name='vreq_reg', dtype='uint64', attrs={'ori_shape':[Cout]})

            requant_res = ascend_requant_compute(conv_res, vreq_reg, None, relu_flag=relu_flag)

            y = {"shape": tuple(i.value for i in requant_res.shape)}

            conv_swrite = strided_write_compute(requant_res, y, 1, stride_swrite, "strided_write")
            auto_sch_res = AutoScheduleOp(conv_swrite)
            sch = auto_schedule(conv_swrite)

        kernel_name = "conv_requant"

        if bias != None:
            tensor_list = [Data_ori, Weight, bias_tensor, vreq_reg, conv_swrite]
            fusion_type = 27
        else:
            tensor_list = [Data_ori, Weight, vreq_reg, conv_swrite]
            fusion_type = 26
        assert auto_sch_res.fusion_type == fusion_type
        config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    """
    The UT for conv vector fused
    """

    cce_conf.te_set_version("Ascend710")

    print("-----------------------------------------------------")
    print("[ v200 conv_sread_swrite UNITTEST START ]")

    print("[ v200 conv_bias_swrite ]")
    conv_bias_swrite()

    print("[ v200 conv_swrite ]")
    conv_swrite()

    print("[ v200 sread_conv_bias_swrite ]")
    sread_conv_bias_swrite()

    print("[ v200 sread_conv_swrite ]")
    sread_conv_swrite()

    print("[ v200 conv_bias_relu_swrite ]")
    conv_bias_relu_swrite()

    print("[ v200 conv_relu_swrite ]")
    conv_relu_swrite()

    print("[ v200 sread_conv_bias_relu_swrite ]")
    sread_conv_bias_relu_swrite()

    print("[ v200 sread_conv_relu_swrite ]")
    sread_conv_relu_swrite()

    print("[ v200 sread_conv_bias_dequant_swrite ]")
    sread_conv_bias_dequant_swrite()

    print("[ v200 sread_conv_dequant_swrite ]")
    sread_conv_dequant_swrite()

    print("[ v200 conv_bias_dequant_swrite ]")
    conv_bias_dequant_swrite()

    print("[ v200 conv_dequant_swrite ]")
    conv_dequant_swrite()

    print("[ v200 stridedread_strideh_opti_int8 ]")
    stridedread_strideh_opti_int8()
    print("[ v200 stridedread_strideh_opti_fp16 ]")
    stridedread_strideh_opti_fp16()

    print("[ v200 stridedread_al1_load2d ]")
    stridedread_al1_load2d()

    print("[ v200 conv_requant_relu_swrite ]")
    conv_requant_relu_swrite()

    print("[ v200 conv_requant_swrite ]")
    conv_requant_swrite()

    print("[ v200 sread_conv_requant_relu_swrite ]")
    sread_conv_requant_relu_swrite()

    print("[ v200 sread_conv_requant_swrite ]")
    sread_conv_requant_swrite()

    cce_conf.te_set_version("Ascend310")

print("adding Conv2D V200 sread swrite testcases")
ut_case.add_cust_test_func(test_func=TestConvSreadSwriteConf)
