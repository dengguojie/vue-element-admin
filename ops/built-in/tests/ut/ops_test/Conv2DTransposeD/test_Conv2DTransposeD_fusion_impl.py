#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the conv2DTransposeD test
"""
import sys

import conv2d_transpose_ut_testcase
import te.lang.cce as tbe
import util_for_conv2d_transpose as util
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_quant import ascend_quant_compute
from impl.ascend_requant import ascend_requant_compute
from impl.conv2d_transpose_d import conv2d_transpose_d_compute
from impl.leaky_relu import leaky_relu_compute
from op_test_frame.ut import OpUT
from te import tvm

ut_case = OpUT("Conv2DTransposeD", "impl.conv2d_transpose_d", "conv2d_transpose_d")
DEBUG_MODE = False


def _gen_kernel_name(dedy_shape, w_shape, dx_shape, strides, data_flow):
    dedy_shape_info = "_".join([str(i) for i in dedy_shape])
    w_shape_info = "_".join([str(i) for i in w_shape])
    dx_shape_info = "_".join([str(i) for i in dx_shape])
    stride_shape_info = "_".join([str(i) for i in strides])

    kernel_name = "Conv2DTransposeD_dy_{}_w_{}_x_{}_stride_{}_data_flow_{}".format(
        dedy_shape_info, w_shape_info, dx_shape_info, stride_shape_info, data_flow
    )
    return kernel_name


def _gen_conv2dtranpose_node(
    dedy_shape,
    filter_shape,
    dedx_shape,
    padding,
    stride,
    dilations,
    bias_flag,
    case_dtype,
):
    """
    the conv2dTranspose node
    """

    if case_dtype == "int8":
        dedx_dtype = "int32"
    else:
        dedx_dtype = "float16"

    dedy_tensor = tvm.placeholder(
        util.shape_4d_to_5hd(dedy_shape, case_dtype, "NCHW"),
        name="dedy",
        dtype=case_dtype,
        attrs={"ori_shape": dedy_shape, "ori_format": "NCHW"},
    )
    weight_tensor = tvm.placeholder(
        util.shape_4d_to_fz(filter_shape, case_dtype, "NCHW"),
        name="filter",
        dtype=case_dtype,
        attrs={
            "ori_shape": util.get_ori_shape(filter_shape, case_dtype, "NCHW"),
            "ori_format": "NCHW",
        },
    )
    dedx_list = {
        "ori_shape": dedx_shape,
        "shape": util.shape_4d_to_5hd(dedx_shape, case_dtype, "NCHW"),
        "dtype": dedx_dtype,
        "ori_format": "NCHW",
        "format": "NC1HWC0",
    }
    tensor_list = [dedy_tensor, weight_tensor]
    if bias_flag:
        bias_tensor = tvm.placeholder(
            (util.align(filter_shape[1], 16),), name="tensor_bias", dtype=dedx_dtype
        )
        tensor_list.append(bias_tensor)
    else:
        bias_tensor = None

    dedx_tensor = conv2d_transpose_d_compute(
        dedy_tensor,
        weight_tensor,
        bias_tensor,
        None,
        dedx_list,
        dedx_shape,
        stride,
        padding,
        dilations,
        1,
        "NCHW",
        (0, 0, 0, 0),
        0,
        "conv2d_transpose_d",
    )
    return dedx_tensor, tensor_list


def _test_conv2d_transpose_fusion(
    soc_version,
    case_dtype,
    dedy_shape,
    filter_shape,
    dedx_shape,
    padding,
    stride,
    dilations,
    bias_flag,
    fusion_para,
):
    """
    the fusion test for conv2dTranspose
    """

    def _test_conv2d_transpose_ubfusion(test_arg):
        with tvm.target.cce():
            # gen conv2dtranspose
            out, tensor_list = _gen_conv2dtranpose_node(
                dedy_shape,
                filter_shape,
                dedx_shape,
                padding,
                stride,
                dilations,
                bias_flag,
                case_dtype,
            )
            channel_in = dedx_shape[1]
            if fusion_para[0] == "relu":
                out = leaky_relu_compute(out, None)
            elif fusion_para[0] in ("dequant", "quant"):
                if fusion_para[2]:
                    shape_deq = (1, util.ceil(channel_in, 16), 1, 1, 16)
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
                    shape_deq = (1, util.ceil(channel_in, 16), 1, 1, 16)
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
            if DEBUG_MODE:
                print(soc_version, tensor_list)
            sch = tbe.auto_schedule(out)
            config = {
                "print_ir": False,
                "need_build": True,
                "name": _gen_kernel_name(
                    dedy_shape, filter_shape, dedx_shape, stride, fusion_para[0]
                ),
                "tensor_list": tensor_list,
            }

            tbe.cce_build_code(sch, config)

    return _test_conv2d_transpose_ubfusion


def _gen_conv2d_transpose_op_fusion_case():
    for fusion_case in conv2d_transpose_ut_testcase.conv2d_transpose_ut_fusion_case:
        ut_case.add_cust_test_func(
            fusion_case[0], test_func=_test_conv2d_transpose_fusion(*fusion_case)
        )


_gen_conv2d_transpose_op_fusion_case()


def _test_conv2d_transpose_fusion_exception(
    soc_version,
    case_dtype,
    dedy_shape,
    filter_shape,
    dedx_shape,
    padding,
    stride,
    dilations,
    bias_flag,
    fusion_para,
):
    """
    the fusion test for conv2dTranspose
    """

    def _test_conv2d_transpose_ubfusion_exception(test_arg):
        with tvm.target.cce():
            # gen conv2dtranspose
            out, tensor_list = _gen_conv2dtranpose_node(
                dedy_shape,
                filter_shape,
                dedx_shape,
                padding,
                stride,
                dilations,
                bias_flag,
                case_dtype,
            )
            channel_in = dedx_shape[1]
            if fusion_para[2]:
                shape_deq = (1, util.ceil(channel_in, 16), 1, 1, 16)
            else:
                shape_deq = (1, 1, 1, 1, 16)
            deq_tensor = tvm.placeholder(
                shape_deq,
                name="deq_scale",
                dtype="float16",
                attrs={"ori_shape": [channel_in if fusion_para[2] else 1]},
            )
            tensor_list.append(deq_tensor)
            out = ascend_dequant_compute(out, deq_tensor, None, fusion_para[1], fusion_para[3])
            out = ascend_quant_compute(out, None, fusion_para[5], fusion_para[6], fusion_para[4], fusion_para[7])
            tensor_list.append(out)
            try:
                sch = tbe.auto_schedule(out)
                config = {
                    "print_ir": False,
                    "need_build": True,
                    "name": _gen_kernel_name(dedy_shape, filter_shape, dedx_shape, stride, fusion_para[0]),
                    "tensor_list": tensor_list,
                }
                tbe.cce_build_code(sch, config)
            except RuntimeError as e:
                print(e)
                print('_gen_conv2d_transpose_op_fusion_exception_case success')

    return _test_conv2d_transpose_ubfusion_exception


def _gen_conv2d_transpose_op_fusion_exception_case():
    for fusion_case in conv2d_transpose_ut_testcase.conv2d_transpose_ut_fusion_exception_case:
        ut_case.add_cust_test_func(fusion_case[0], test_func=_test_conv2d_transpose_fusion_exception(*fusion_case))


_gen_conv2d_transpose_op_fusion_exception_case()


if __name__ == "__main__":
    ut_case.run(["Ascend910", "Ascend710", "Ascend310"])
    sys.exit(0)
