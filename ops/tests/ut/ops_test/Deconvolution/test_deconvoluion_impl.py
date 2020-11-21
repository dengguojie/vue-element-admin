#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
from copy import deepcopy
from math import ceil as math_ceil

from deconvolution_ut_testcase import deconvolution_ut_case
from deconvolution_ut_testcase import deconvolution_ut_fusion_case
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_quant import ascend_quant_compute
from impl.ascend_requant import ascend_requant_compute
from impl.deconvolution import deconvolution_compute
from impl.leaky_relu import leaky_relu_compute
from op_test_frame.ut import OpUT
from te import tvm
from te.lang.cce import cce_build_code
from te.tvm.target import cce
from topi.generic import auto_schedule

ut_case = OpUT("Deconvolution", "impl.deconvolution", "deconvolution")

"""
the deconvolution test
"""


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


def _shape_nchw_to_fz(shape, dtype):
    batch, channel, height, weight = shape
    block_k = _get_block("k", dtype)
    block_n = _get_block("n", dtype)

    channel1 = _ceil(channel, block_k)
    channel0 = block_n
    batch1 = _ceil(batch, block_n)
    batch0 = block_k
    if dtype == "int8":
        return batch1 * height * weight, channel1, channel0, batch0
    return channel1 * height * weight, batch1, batch0, channel0


def _get_ori_shape(shape, dtype):
    if dtype == "int8":
        return [shape[1], shape[0], shape[2], shape[3]]
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
    expect_result = "success"
    if len(case) == 10:
        expect_result = case[9]

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
        "ori_shape": _get_ori_shape(filter_shape, case_dtype),
        "shape": _shape_nchw_to_fz(filter_shape, case_dtype),
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
            1,
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
        _shape_nchw_to_fz(filter_shape, case_dtype),
        name="filter",
        dtype=case_dtype,
        attrs={
            "ori_shape": _get_ori_shape(filter_shape, case_dtype),
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
            elif fusion_para[0] in ("dequant", "quant"):
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
                if fusion_para[0] == "quant":
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

if __name__ == "__main__":
    ut_case.run()
    sys.exit(0)
