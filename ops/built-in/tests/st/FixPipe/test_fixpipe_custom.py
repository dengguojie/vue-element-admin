#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.fix_pipe import fixpipe_compute
from impl.fixpipe_op.fixpipe_factory import FixpipeFactory
from impl.fixpipe_op.fixpipe_conv2d_backprop_filter import FixpipeConv2dBackpropFilter
from te import tvm
from tbe.dsl import auto_schedule
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.deconvolution import deconvolution_compute
from te.tvm.target import cce
from impl.fixpipe_op.fixpipe_conv2d_backprop_input import FixpipeConv2dBackpropInput
from impl.conv2d_backprop_filter_d import conv2d_backprop_filter_compute
from te.tvm.target import cce

vals = {
    ("CORE_NUM", ): 48,
    ("CUBE_VECTOR_SPLIT",): True,
    ("UB_SIZE", ): 196608,
    ("L0A_SIZE", ): 65536,
    ("L0B_SIZE", ): 65536,
    ("L1_SIZE", ): 524288,
    ("L0C_SIZE", ): 131072,
    ("Compiler_arch",): "dav-c220-cube",
    ("AICORE_TYPE",): "AiCore",
    ("SOC_VERSION",): "Ascend920A",
    ("Intrinsic_fix_pipe_unit_list",): True,
    ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True,
    ("Intrinsic_fix_pipe_l0c2ub",) : True,
    ("Intrinsic_fix_pipe_l0c2out",) : True,
    ("Intrinsic_data_move_l0c2ub",) : True,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : True,
    ("Intrinsic_mmad", "f162f32",) : True,
    ("CUBE_VECTOR_SPLIT",) : False,
}

def get_soc_mock(*args):
    return vals[args]

def test_conv2d_bp_filter_fixpipe_0():
    with cce():
        fmap_tensor = tvm.placeholder((1, 2, 28, 28, 16), name="fmap", dtype="float16", attrs={"ori_shape": (1, 32, 28, 28), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedy_tensor = tvm.placeholder((1, 1, 26, 26, 16), name="dedy", dtype="float16", attrs={"ori_shape": (1, 16, 26, 26), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedw = {"shape": (16, 32, 3, 3), "dtype": "float32", "ori_shape": (16, 32, 3, 3), "format": "NCHW", "ori_format": "NCHW"}
        filter_size = (16, 32, 3, 3)
        strides = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)
        dilations=(1, 1, 1, 1)
        groups = 1
        data_format = "NCHW"
        dedw_tensor = conv2d_backprop_filter_compute(fmap_tensor, dedy_tensor, dedw, filter_size, strides, pads, dilations, groups, data_format)
        output_dict = {"shape": (16, 3, 3, 32), "dtype": "float32", "format": "NHWC"}
        fixpipe = FixpipeConv2dBackpropFilter("conv2d_backprop_filter", dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        res = fixpipe.fixpipe_compute()
        tensor_list = [fmap_tensor, dedy_tensor, res]
        # sch = auto_schedule(res)

def test_conv2d_bp_filter_fixpipe_1():
    with cce():
        fmap_tensor = tvm.placeholder((1, 4, 28, 28, 8), name="fmap", dtype="float32", attrs={"ori_shape": (1, 32, 28, 28), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedy_tensor = tvm.placeholder((1, 2, 26, 26, 8), name="dedy", dtype="float32", attrs={"ori_shape": (1, 16, 26, 26), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedw = {"shape": (16, 32, 3, 3), "dtype": "float32", "ori_shape": (16, 32, 3, 3), "format": "NCHW", "ori_format": "NCHW"}
        filter_size = (16, 32, 3, 3)
        strides = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)
        dilations=(1, 1, 1, 1)
        groups = 1
        data_format = "NCHW"
        dedw_tensor = conv2d_backprop_filter_compute(fmap_tensor, dedy_tensor, dedw, filter_size, strides, pads, dilations, groups, data_format)
        output_dict = {"shape": (16, 3, 3, 32), "dtype": "float32", "format": "NHWC"}
        fixpipe = FixpipeConv2dBackpropFilter("conv2d_backprop_filter", dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        res = fixpipe.fixpipe_compute()
        tensor_list = [fmap_tensor, dedy_tensor, res]
        # sch = auto_schedule(res)


def test_conv2d_bp_filter_fixpipe_2():
    with cce():
        fmap_tensor = tvm.placeholder((1, 2, 28, 28, 16), name="fmap", dtype="float16", attrs={"ori_shape": (1, 32, 28, 28), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedy_tensor = tvm.placeholder((1, 1, 26, 26, 16), name="dedy", dtype="float16", attrs={"ori_shape": (1, 16, 26, 26), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedw = {"shape": (16, 32, 3, 3), "dtype": "float32", "ori_shape": (16, 32, 3, 3), "format": "NCHW", "ori_format": "NCHW"}
        filter_size = (16, 32, 3, 3)
        strides = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)
        dilations=(1, 1, 1, 1)
        groups = 1
        data_format = "NCHW"
        dedw_tensor = conv2d_backprop_filter_compute(fmap_tensor, dedy_tensor, dedw, filter_size, strides, pads, dilations, groups, data_format)
        output_dict = {"shape": (18, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z"}
        fixpipe = FixpipeConv2dBackpropFilter("conv2d_backprop_filter", dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        res = fixpipe.fixpipe_compute()
        tensor_list = [fmap_tensor, dedy_tensor, res]
        # sch = auto_schedule(res)

def test_conv2d_dx_fixpie_deconv_eltwise_0():
    filter_frac = (18, 2, 16, 16)
    out_shape_5hd = (16, 2, 2, 2, 16)
    input_size = (16, 32, 4, 4)
    input_size_5hd = (16, 2, 4, 4, 16)
    data_type = "float16"
    with cce():
        weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                attrs={"ori_shape": (32, 32, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
        dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                            attrs={"ori_shape": (16, 32, 2, 2), "dtype":data_type, "ori_format": "NCHW"})
        eltwise_plh = tvm.placeholder((input_size[0], (input_size[1] + 15) // 16, input_size[2] * input_size[3], 16),
                                        name="fixpipe_eltwise", dtype="float16",
                                        attrs={"ori_format": "NCHW", "ori_shape": input_size})
        y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW", "shape": input_size_5hd, "format": "NC1HWC0"}
        out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                kernel_name="deconvolution")
        fixpipe_op = FixpipeConv2dBackpropInput("conv2d_backprop_input", out, eltwise_plh, None, None, None, None, None, None, None, None, y, [], [], "None")
        res = fixpipe_op.fixpipe_compute()
        sch = auto_schedule(res)

def test_conv2d_dx_fixpie_deconv_dequant():
    filter_frac = (18, 2, 16, 16)
    out_shape_5hd = (16, 2, 2, 2, 16)
    input_size = (16, 32, 4, 4)
    input_size_5hd = (16, 2, 4, 4, 16)
    data_type = "float16"
    with cce():
        weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                attrs={"ori_shape": (32, 32, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
        dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                            attrs={"ori_shape": (16, 32, 2, 2), "dtype":data_type, "ori_format": "NCHW"})
        bias = tvm.placeholder((32,), name="bias", dtype="float32",
                            attrs={"ori_shape": (32,), "dtype":"float32", "ori_format": "ND"})
        y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW", "shape": input_size_5hd, "format": "NC1HWC0"}
        deq = tvm.placeholder((1, 2, 1, 1, 16), name='deq', dtype="uint64", attrs={"ori_shape": (32, ), "format": "NC1HWC0", "ori_format": "ND"})
        out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                kernel_name="deconvolution")
        fixpie_op = FixpipeConv2dBackpropInput("conv2d_backprop_input", out, None, deq, None, None, None, None, None, None, None, y, [], [], "")
        res = fixpie_op.fixpipe_compute()
        sch = auto_schedule(res)

def test_conv2d_dx_fixpie_deconv_nz2nd():
    filter_frac = (18, 2, 16, 16)
    out_shape_5hd = (16, 2, 2, 2, 16)
    input_size = (16, 32, 4, 4)
    input_size_nhwc = (input_size[0], input_size[2], input_size[3], input_size[1])
    data_type = "float16"
    with cce():
        weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                attrs={"ori_shape": (32, 32, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
        dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                            attrs={"ori_shape": (16, 32, 2, 2), "dtype":data_type, "ori_format": "NCHW"})
        bias = tvm.placeholder((32,), name="bias", dtype="float32",
                            attrs={"ori_shape": (32,), "dtype":"float32", "ori_format": "ND"})
        y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW",
            "shape": input_size_nhwc, "format": "NHWC"}
        out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                kernel_name="deconvolution")
        fixpie_op = FixpipeConv2dBackpropInput("conv2d_backprop_input", out, None, None, None, None, None, None, None, None, None, y, [], [], "")
        res = fixpie_op.fixpipe_compute()
        sch = auto_schedule(res)


def test_fixpipe_cases():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=get_soc_mock)):
        with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
            with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=get_soc_mock)):
                test_conv2d_bp_filter_fixpipe_0()
                test_conv2d_bp_filter_fixpipe_1()
                test_conv2d_bp_filter_fixpipe_2()
                test_conv2d_dx_fixpie_deconv_eltwise_0()
                test_conv2d_dx_fixpie_deconv_dequant()
                test_conv2d_dx_fixpie_deconv_nz2nd()


if __name__ == '__main__':
    test_fixpipe_cases()