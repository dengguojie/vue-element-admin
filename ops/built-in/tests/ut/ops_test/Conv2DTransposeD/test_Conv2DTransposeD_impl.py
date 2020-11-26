#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the conv2DTransposeD test
"""
import sys

import conv2d_transpose_ut_testcase
import util_for_conv2d_transpose as util
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2DTransposeD", "impl.conv2d_transpose_d", "conv2d_transpose_d")


def _gen_kernel_name(dedy_shape, w_shape, dx_shape, strides, data_flow):
    dedy_shape_info = "_".join([str(i) for i in dedy_shape])
    w_shape_info = "_".join([str(i) for i in w_shape])
    dx_shape_info = "_".join([str(i) for i in dx_shape])
    stride_shape_info = "_".join([str(i) for i in strides])

    kernel_name = "Conv2DTransposeD_dy_{}_w_{}_x_{}_stride_{}_data_flow_{}".format(
        dedy_shape_info, w_shape_info, dx_shape_info, stride_shape_info, data_flow
    )
    return kernel_name


def _gen_conv2d_transpose_case(
    dedy_dtype,
    w_dtype,
    dx_dtype,
    dedy_shape,
    w_shape,
    dx_shape,
    dedy_format,
    w_format,
    dx_format,
    input_size,
    bias_flag,
    stride,
    padding,
    output_padding=(0, 0, 0, 0),
    dilations=(1, 1, 1, 1),
    groups=1,
    expect="success",
    data_flow="default",
):

    kernel_name = _gen_kernel_name(dedy_shape, w_shape, dx_shape, stride, data_flow)

    dedy = {
        "shape": util.shape_4d_to_5hd(dedy_shape, dedy_dtype, dedy_format),
        "dtype": dedy_dtype,
        "format": dedy_format,
        "ori_shape": dedy_shape,
        "ori_format": dedy_format,
    }

    filters = {
        "shape": util.shape_4d_to_fz(w_shape, w_dtype, w_format),
        "dtype": w_dtype,
        "format": w_format,
        "ori_shape": util.get_ori_shape(w_shape, w_dtype, w_format),
        "ori_format": w_format,
    }

    dedx = {
        "shape": util.shape_4d_to_5hd(dx_shape, dx_dtype, dx_format),
        "dtype": dx_dtype,
        "format": dx_format,
        "ori_shape": dx_shape,
        "ori_format": dx_format,
    }

    input_sizes = input_size
    bias = None
    # NCHW
    strides = stride
    padding = util.gen_padding_size(dx_shape, w_shape, padding, stride, dilations)
    data_format = dedy_format
    offset_w = None
    return {
        "params": [
            dedy,
            filters,
            bias,
            offset_w,
            dedx,
            input_sizes,
            strides,
            padding,
            dilations,
            groups,
            dedy_format,
            output_padding,
            0,
        ],
        "case_name": kernel_name,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


def _gen_conv2d_transpose_op_case():
    for test_case in conv2d_transpose_ut_testcase.conv2d_transpose_op_testcase:
        ut_case.add_case(["Ascend910"], _gen_conv2d_transpose_case(*test_case))


_gen_conv2d_transpose_op_case()


if __name__ == "__main__":
    ut_case.run(["Ascend910"])
    sys.exit(0)
