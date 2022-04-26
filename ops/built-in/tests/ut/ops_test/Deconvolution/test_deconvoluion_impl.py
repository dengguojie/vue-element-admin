#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
from copy import deepcopy
from math import ceil as math_ceil

from unittest.mock import MagicMock
from unittest.mock import patch
from deconvolution_ut_testcase import deconvolution_ut_case
from deconvolution_ut_testcase import deconvolution_ut_fusion_case
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_quant import ascend_quant_compute
from impl.ascend_requant import ascend_requant_compute
from impl.deconvolution import deconvolution_compute
from impl.deconvolution import get_op_support_info
from impl.leaky_relu import leaky_relu_compute
from impl.prelu import prelu_compute
from op_test_frame.ut import OpUT
from te import tvm
from te.lang.cce import cce_build_code
from te.tvm.target import cce
from tbe.dsl import auto_schedule
from impl.fix_pipe import fixpipe_compute
from impl.fixpipe_op.fixpipe_conv2d_backprop_input import *

ut_case = OpUT("Deconvolution", "impl.deconvolution", "deconvolution")

"""
the deconvolution test
"""

support_intrinsic_no_fixpipe = {
    ("Intrinsic_fix_pipe_l0c2ub",) : False,
    ("Intrinsic_fix_pipe_l0c2out",) : False,
    ("Intrinsic_data_move_l0c2ub",) : True,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : True,
    ("Intrinsic_mmad", "f162f32",) : False,
    ("CUBE_VECTOR_SPLIT",) : False,
}

support_intrinsic_mix = {
    ("Intrinsic_fix_pipe_l0c2ub",) : True,
    ("Intrinsic_fix_pipe_l0c2out",) : True,
    ("Intrinsic_data_move_l0c2ub",) : True,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : True,
    ("Intrinsic_mmad", "f162f32",) : True,
    ("CUBE_VECTOR_SPLIT",) : True,
    ("Intrinsic_fix_pipe_unit_list",): True,
    ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True
}

def check_intrinsic_mock_mix(*args):
    return support_intrinsic_mix[args]

def check_intrinsic_mock_no_fixpipe(*args):
    return support_intrinsic_no_fixpipe[args]

def _get_block(val, dtype):
    assert val in ("m", "n", "k")
    if val in ("m", "n"):
        return 16
    return 32 if dtype in ["int8", "uint8"] else 16


def _ceil(val, base):
    return math_ceil(val / base)


def _align(val, base):
    return _ceil(val, base) * base


def _shape_nchw_to_nc1hwc0(shape, dtype):
    shape_nc1hwc0 = list(deepcopy(shape))
    block_k = _get_block("k", dtype)
    shape_nc1hwc0[1] = _align(shape_nc1hwc0[1], block_k)
    return (
        shape_nc1hwc0[0],
        shape_nc1hwc0[1] // block_k,
        shape_nc1hwc0[2],
        shape_nc1hwc0[3],
        block_k,
    )


def _shape_nchw_to_fz(shape, dtype, groups):
    batch, channel, height, weight = shape
    block_k = _get_block("k", dtype)
    block_n = _get_block("n", dtype)
    if dtype == "int8":
        batch = batch // groups
    channel1 = _ceil(channel, block_n)
    channel0 = block_n
    batch1 = _ceil(batch, block_k)
    batch0 = block_k
    if dtype == "int8":
        return groups * batch1 * height * weight, channel1, channel0, batch0
    return channel1 * height * weight, batch1, batch0, channel0


def _get_ori_shape(shape, dtype, groups):
    if dtype == "int8":
        return [shape[1] * groups, shape[0] // groups, shape[2], shape[3]]
    return shape


def gen_deconv_case(case):
    """
    the single test for deconv
    """
    (
        case_dtype,
        dedy_shape,
        filter_shape,
        dedx_shape,
        padding,
        stride,
        dilation,
        bias_flag,
    ) = case[1:9]
    groups = 1
    expect_result = "success"
    if len(case) == 10:
        expect_result = case[9]
    if len(case) == 11:
        groups = case[10]

    if case_dtype == "int8":
        dedx_dtype = "int32"
    else:
        dedx_dtype = "float16"

    dedy = {
        "ori_shape": dedy_shape,
        "shape": _shape_nchw_to_nc1hwc0(dedy_shape, case_dtype),
        "dtype": case_dtype,
        "ori_format": "NCHW",
        "format": "NC1HWC0",
    }
    filters = {
        "ori_shape": _get_ori_shape(filter_shape, case_dtype, groups),
        "shape": _shape_nchw_to_fz(filter_shape, case_dtype, groups),
        "dtype": case_dtype,
        "ori_format": "NCHW",
        "format": "FRACTAL_Z",
    }
    dedx = {
        "ori_shape": dedx_shape,
        "shape": _shape_nchw_to_nc1hwc0(dedx_shape, dedx_dtype),
        "dtype": dedx_dtype,
        "ori_format": "NCHW",
        "format": "NC1HWC0",
    }
    bias = None
    if bias_flag:
        bias = {
            "ori_shape": (filter_shape[1],),
            "shape": (_align(filter_shape[1], 16),),
            "ori_format": "ND",
            "format": "ND",
            "dtype": dedx_dtype,
        }

    return {
        "params": [
            dedy,
            filters,
            bias,
            None,
            dedx,
            stride,
            padding,
            [1, 1] + list(dilation),
            groups,
            "NCHW",
            0,
        ],
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True,
    }


def _get_deconv_node(case):
    """
    the deconv node
    """
    (
        case_dtype,
        dedy_shape,
        filter_shape,
        dedx_shape,
    ) = case[0:4]
    groups = 1
    if case_dtype == "int8":
        dedx_dtype = "int32"
    else:
        dedx_dtype = "float16"

    dedy = tvm.placeholder(
        _shape_nchw_to_nc1hwc0(dedy_shape, case_dtype),
        name="dedy",
        dtype=case_dtype,
        attrs={"ori_shape": dedy_shape, "ori_format": "NCHW"},
    )

    weight = tvm.placeholder(
        _shape_nchw_to_fz(filter_shape, case_dtype, groups),
        name="filter",
        dtype=case_dtype,
        attrs={
            "ori_shape": _get_ori_shape(filter_shape, case_dtype, groups),
            "ori_format": "NCHW",
        },
    )
    dedx_list = {
        "ori_shape": dedx_shape,
        "shape": _shape_nchw_to_nc1hwc0(dedx_shape, dedx_dtype),
        "dtype": dedx_dtype,
        "ori_format": "NCHW",
        "format": "NC1HWC0",
    }
    tensor_list = [dedy, weight]
    if case[7]:
        bias_tensor = tvm.placeholder(
            (_align(filter_shape[1], 16),), name="tensor_bias", dtype=dedx_dtype
        )
        tensor_list.append(bias_tensor)
    else:
        bias_tensor = None

    dedx = deconvolution_compute(
        dedy,
        weight,
        bias_tensor,
        None,
        dedx_list,
        [1, 1] + list(case[5]),
        case[4],
        [1, 1] + list(case[6]),
        data_format="NCHW",
    )
    return dedx, tensor_list


def _get_kernelname(para):
    kernel_name = "deconv_"
    if para[0] == "relu":
        kernel_name += para[0]
    elif para[0] in ("dequant", "quant"):
        kernel_name += "dequant_sqrt_{}_vector_{}_relu_{}".format(
            str(para[1]), str(para[2]), str(para[3])
        )
        if para[0] == "quant":
            str_scalar = str(para[5]).replace(".", "p").replace("-", "N")
            str_offset = str(para[6]).replace(".", "p").replace("-", "N")
            kernel_name += "_quant_sqrt_{}_scalar_{}_offset_{}".format(
                str(para[4]), str_scalar, str_offset
            )
    elif para[0] == "requant":
        kernel_name += "_requant_vector_{}_relu_{}".format(str(para[1]), str(para[2]))
    return kernel_name


def test_deconvolution_fusion(case):
    """
    the fusion test for deconv
    """

    def test_deconvolution_ubfusion(test_arg):

        fusion_para = case[9]

        with cce():
            # deconv
            out, tensor_list = _get_deconv_node(case[1:9])
            channel_in = case[4][1]
            if fusion_para[0] == "relu":
                out = leaky_relu_compute(out, None)
            elif fusion_para[0] == "prelu":
                weight_shape = [1] * len(out.shape)
                weight_shape[1] = out.shape[1]
                weight_shape[-1] = out.shape[-1]
                weight_input = tvm.placeholder(weight_shape, name='weight_input', dtype="float16", )
                tensor_list.append(weight_input)
                out = prelu_compute(out, weight_input, None)
            elif fusion_para[0] in ("dequant", "quant", "dequant_prelu",
                                    "dequant_leckyrelu_quant", "dequant_prelu_quant_double"):
                if fusion_para[2]:
                    shape_deq = (1, _ceil(channel_in, 16), 1, 1, 16)
                else:
                    shape_deq = (1, 1, 1, 1, 16)
                deq_tensor = tvm.placeholder(
                    shape_deq,
                    name="deq_scale",
                    dtype="float16",
                    attrs={"ori_shape": [channel_in if fusion_para[2] else 1]},
                )
                tensor_list.append(deq_tensor)
                out = ascend_dequant_compute(
                    out, deq_tensor, None, fusion_para[1], fusion_para[3]
                )

                if fusion_para[0] in ("dequant_prelu", "dequant_prelu_quant_double"):
                    weight_shape = [1] * len(out.shape)
                    weight_shape[1] = out.shape[1]
                    weight_shape[-1] = out.shape[-1]
                    weight_input = tvm.placeholder(weight_shape, name='weight_input', dtype="float16", )
                    tensor_list.append(weight_input)
                    out = prelu_compute(out, weight_input, None)
                    prelu_out = out
                elif fusion_para[0] == "dequant_leckyrelu_quant":
                    out = leaky_relu_compute(out, None, negative_slope=0.1)

                if fusion_para[0] in ("quant", "dequant_leckyrelu_quant", "dequant_prelu_quant_double"):
                    out = ascend_quant_compute(
                        out, None, fusion_para[5], fusion_para[6], fusion_para[4]
                    )
            elif fusion_para[0] == "requant":
                if fusion_para[1]:
                    shape_deq = (1, _ceil(channel_in, 16), 1, 1, 16)
                else:
                    shape_deq = (1, 1, 1, 1, 16)
                deq_tensor = tvm.placeholder(
                    shape_deq,
                    name="deq_scale",
                    dtype="uint64",
                    attrs={"ori_shape": [channel_in if fusion_para[1] else 1]},
                )
                tensor_list.append(deq_tensor)
                out = ascend_requant_compute(out, deq_tensor, None, fusion_para[2])
            if fusion_para[0] == "dequant_prelu_quant_double":
                out = [prelu_out, out]
                tensor_list.extend(out)
            else:
                tensor_list.append(out)
            sch = auto_schedule(out)
            config = {
                "print_ir": False,
                "need_build": True,
                "name": _get_kernelname(fusion_para),
                "tensor_list": tensor_list,
            }

            cce_build_code(sch, config)

    return test_deconvolution_ubfusion


def test_op_check_supported(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_no_fixpipe)):
        from impl.deconvolution import check_supported
        x = {"ori_shape": (1, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
        weight = {"ori_shape": (32, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW"}
        bias = None
        y = {"ori_shape": (1, 16, 5, 5), "dtype": "float16", "ori_format": "NCHW"}

        check_supported(x, weight, bias, None, y, (1, 1, 2, 2), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")


def test_op_group_requant(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_no_fixpipe)):
        x = {"ori_shape": (1, 192, 48, 80), "dtype": "int8", "ori_format": "NCHW", "shape": (1, 6, 48, 80, 32), "format":"NC1HWC0"}
        weight = {"ori_shape": (144, 64, 3, 3), "dtype": "int8", "ori_format": "NCHW", "shape": (54, 3, 16, 32), "format": "FRACTAL_NZ"}
        bias = None
        y = {"ori_shape": (1, 144, 48, 80), "dtype": "int8", "ori_format": "NCHW",  "shape": (1, 5, 48, 80, 32), "format":"NC1HWC0"}
        from impl.deconvolution import deconvolution
        try:
            deconvolution(x, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=10, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
        except RuntimeError as e:
            print(e)
            pass

def test_op_mix_case1_mock(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (144, 16, 16, 16)
        out_shape_5hd = (2, 16, 14, 14, 16)
        input_size = (2, 256, 28, 28)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (256, 256, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 256, 14, 14), "dtype":data_type, "ori_format": "NCHW"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 2, 2), (0, 1, 0, 1),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                    kernel_name="deconvolution")
            sch = auto_schedule(out)

def test_op_mix_case2_mock(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (2, 2, 16, 16)
        out_shape_5hd = (2, 2, 7, 7, 16)
        input_size = (2, 32, 14, 14)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (32, 32, 1, 1), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 32, 7, 7), "dtype":data_type, "ori_format": "NCHW"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 2, 2), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
            sch = auto_schedule(out)


def test_op_mix_case3_mock(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (2, 2, 16, 16)
        out_shape_5hd = (2, 2, 7, 7, 16)
        input_size = (2, 32, 7, 7)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (32, 32, 1, 1), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 32, 7, 7), "dtype":data_type, "ori_format": "NCHW"})
            bias = tvm.placeholder((32,), name="bias", dtype="float32",
                                attrs={"ori_shape": (32,), "dtype":data_type, "ori_format": "ND"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
            sch = auto_schedule(out)

def test_op_mix_case4_mock(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (144, 16, 16, 16)
        out_shape_5hd = (2, 16, 14, 14, 16)
        input_size = (2, 256, 28, 28)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (256, 256, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 256, 14, 14), "dtype":data_type, "ori_format": "NCHW"})
            bias = tvm.placeholder((32,), name="bias", dtype="float32",
                                attrs={"ori_shape": (256,), "dtype":data_type, "ori_format": "ND"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 2, 2), (0, 1, 0, 1),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                    kernel_name="deconvolution")
            sch = auto_schedule(out)

def test_op_compute_int8(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_no_fixpipe)):
        x = tvm.placeholder(
            (1, 6, 48, 80, 32),
            name="x",
            dtype="int8",
            attrs={"ori_shape": (1, 192, 48, 80), "ori_format": "NCHW"},
        )
        weight_tensor = tvm.placeholder(
            (54, 3, 16, 32),
            name="filter",
            dtype="int8",
            attrs={
                "ori_shape": (144, 64, 3, 3),
                "ori_format": "NCHW",
            },
        )
        dedx_list = {
            "ori_shape": (1,144,48,80),
            "shape": (1,5,48,80,32),
            "dtype": "int8",
            "ori_format": "NCHW",
            "format": "NC1HWC0",
        }
        try:
            deconvolution_compute(
            x, weight_tensor, None, None, dedx_list,
            (1, 1, 1, 1), (0, 0, 0, 0),
            dilations=(1, 1, 1, 1), groups=10,
            )
        except RuntimeError as e:
            print(e)
            pass


for case_ut in deconvolution_ut_case:
    print("==========add case for deconvlution===============")
    print("the case is :", case_ut)
    ut_case.add_case(case_ut[0], gen_deconv_case(case_ut))

for fusion_case in deconvolution_ut_fusion_case:
    print("==========add case for deconvlution fusion===============")
    print("the fusion_case is ", fusion_case)
    ut_case.add_cust_test_func(
        fusion_case[0], test_func=test_deconvolution_fusion(fusion_case)
    )

def test_split_deconvolution(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_no_fixpipe)):
        x = {"ori_shape": (1, 192, 48, 80), "dtype": "int8", "ori_format": "NCHW", "shape": (1, 6, 48, 80, 32), "format":"NC1HWC0"}
        weight = {"ori_shape": (144, 64, 3, 3), "dtype": "int8", "ori_format": "NCHW", "shape": (54, 3, 16, 32), "format": "FRACTAL_NZ"}
        bias = None
        y = {"ori_shape": (1, 144, 48, 80), "dtype": "int8", "ori_format": "NCHW",  "shape": (1, 5, 48, 80, 32), "format":"NC1HWC0"}
        get_op_support_info(x, weight, bias, None, y, (1, 1), (0, 0, 0, 0))

def test_fixpipe_nz2nd(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
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
                y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW",
                    "shape": input_size_nhwc, "format": "NHWC"}
                out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
                fixpie_op = FixpipeConv2dBackpropInput("conv2d_backprop_input", out, None, None, None, None, None, None, None, None, None, y, [], [], "")
                res = fixpie_op.fixpipe_compute()
                sch = auto_schedule(res)

def test_fixpipe_eltwise(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
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
                fixpie_op = FixpipeConv2dBackpropInput("conv2d_backprop_input", out, eltwise_plh, None, None, None, None, None, None, None, None, y, [], [], "ADD")
                res = fixpie_op.fixpipe_compute()
                sch = auto_schedule(res)


def test_fixpipe_dequant(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
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
                                    attrs={"ori_shape": (32,), "dtype":data_type, "ori_format": "ND"})
                y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW", "shape": input_size_5hd, "format": "NC1HWC0"}
                deq = tvm.placeholder((1, 2, 1, 1, 16), name='deq', dtype="uint64", attrs={"ori_shape": (32, ), "format": "NC1HWC0", "ori_format": "ND"})
                out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
                fixpie_op = FixpipeConv2dBackpropInput("conv2d_backprop_input", out, None, deq, None, None, None, None, None, None, None, y, [], [], "")
                res = fixpie_op.fixpipe_compute()
                sch = auto_schedule(res)


def test_bt_fp162fp32(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
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
                bias = tvm.placeholder((32,), name="bias", dtype="float16",
                                    attrs={"ori_shape": (32,), "dtype":data_type, "ori_format": "ND"})
                y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW", "shape": input_size_5hd, "format": "NC1HWC0"}
                deq = tvm.placeholder((1, 2, 1, 1, 16), name='deq', dtype="uint64", attrs={"ori_shape": (32, ), "format": "NC1HWC0", "ori_format": "ND"})
                out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
                fixpie_op = FixpipeConv2dBackpropInput("conv2d_backprop_input", out, None, deq, None, None, None, None, None, None, None, y, [], [], "")
                res = fixpie_op.fixpipe_compute()
                sch = auto_schedule(res)

ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_op_group_requant)
ut_case.add_cust_test_func(test_func=test_op_compute_int8)
ut_case.add_cust_test_func(test_func=test_split_deconvolution)
ut_case.add_cust_test_func(test_func=test_op_mix_case1_mock)
ut_case.add_cust_test_func(test_func=test_op_mix_case2_mock)
ut_case.add_cust_test_func(test_func=test_op_mix_case3_mock)
ut_case.add_cust_test_func(test_func=test_op_mix_case4_mock)
ut_case.add_cust_test_func(test_func=test_fixpipe_nz2nd)
ut_case.add_cust_test_func(test_func=test_fixpipe_eltwise)
ut_case.add_cust_test_func(test_func=test_fixpipe_dequant)


if __name__ == "__main__":
    ut_case.run("Ascend310")
    sys.exit(0)
