#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the conv2DTransposeD test
"""
import sys

import conv2d_transpose_ut_testcase
import util_for_conv2d_transpose as util
from op_test_frame.ut import OpUT
from impl.conv2d_transpose_d import get_op_support_info
from impl.conv2d_transpose_d import conv2d_transpose_d
from impl.conv2d_transpose_d import conv2d_transpose_d_compute
from te import tvm

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
    if bias_flag:
        bias = {
            "shape": (util.align(input_sizes[1], 16),),
            "dtype": dx_dtype,
            "format": dx_format,
            "ori_shape": (input_sizes[1],),
            "ori_format": dx_format,
        }
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
        ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], _gen_conv2d_transpose_case(*test_case))


def test_op_check_supported(test_arg):
    from impl.conv2d_transpose_d import check_supported
    x = {"ori_shape": (1, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
    filter = {"ori_shape": (32, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW"}
    bias = None
    y = {"ori_shape": (1, 16, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    input_size = (1, 16, 5, 5)
    check_supported(x, filter, bias, None, y, input_size, (1, 1, 2, 2), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", output_padding=(0, 0, 0, 0),
                    offset_x=0, kernel_name="deconvolution")


def test_split_transposed(test_arg):
    x = {"format": "NC1HWC0","ori_format": "NCHW", "dtype": "float16", "shape": (1, 1, 3, 3, 16), "ori_shape": (1, 16, 3, 3)}
    filter = {"format": "FRACTAL_NZ","ori_format": "NCHW", "dtype": "float16", "shape": (1, 1, 16, 16), "ori_shape": (16, 16, 1, 1)}
    y = {"format": "NC1HWC0","ori_format": "NCHW", "dtype": "float16", "shape": (1, 1, 3, 3, 16), "ori_shape": (1, 16, 3, 3)}
    get_op_support_info(x, filter, None, None, y, (1, 16, 3, 3), (1, 1, 1, 1), (0, 0, 0, 0), data_format="NCHW")


def test_op_group_int8(test_arg):
    x = {"ori_shape": (1, 192, 48, 80), "dtype": "int8", "ori_format": "NCHW","shape": (1, 6, 48, 80, 32), "format":"NC1HWC0"}
    weight = {"ori_shape": (144, 64, 3, 3), "dtype": "int8", "ori_format": "NCHW", "shape": (54, 3, 16, 32), "format": "FRACTAL_NZ"}
    bias = None
    y = {"ori_shape": (1, 144, 48, 32), "dtype": "int8", "ori_format": "NCHW", "shape": (1, 5, 48, 80, 32), "format":"NC1HWC0"}

    try:
        conv2d_transpose_d(x, weight, bias, None, y, (1, 144, 48, 32), (1, 1, 1, 1), (0, 0, 0, 0),
                           dilations=(1, 1, 1, 1), groups=10, data_format="NCHW", offset_x=0,
                           kernel_name="deconvolution")
    except RuntimeError as e:
        print(e)
        pass


def test_op_compute_int8(test_arg):
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
        conv2d_transpose_d_compute(
        x, weight_tensor, None, None, dedx_list,
        (1, 144, 48, 80),
        (1, 1, 1, 1), (0, 0, 0, 0),
        dilations=(1, 1, 1, 1), groups=10,
        )
    except RuntimeError as e:
        print(e)
        pass

_gen_conv2d_transpose_op_case()
ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_split_transposed)
ut_case.add_cust_test_func(test_func=test_op_group_int8)
ut_case.add_cust_test_func(test_func=test_op_compute_int8)


if __name__ == "__main__":
    ut_case.run(["Ascend310", "Ascend710", "Ascend910"])
    sys.exit(0)
