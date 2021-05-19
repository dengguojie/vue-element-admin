#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys

import conv2d_bp_filter_ut_testcase
import util_for_conv2d_bp_filter as util
from op_test_frame.ut import OpUT
from impl.conv2d_backprop_filter_d import get_op_support_info

ut_case = OpUT(
    "conv2d_backprop_filter_d",
    "impl.conv2d_backprop_filter_d",
    "conv2d_backprop_filter_d",
)
DEBUG_MODE = False


def _gen_kernel_name(dedy_shape, w_shape, dx_shape, strides, data_flow):
    dedy_shape_info = "_".join([str(i) for i in dedy_shape])
    w_shape_info = "_".join([str(i) for i in w_shape])
    dx_shape_info = "_".join([str(i) for i in dx_shape])
    stride_shape_info = "_".join([str(i) for i in strides])

    kernel_name = "Conv2DBackpropFilterD_dy_{}_w_{}_x_{}_stride_{}_data_flow_{}".format(
        dedy_shape_info, w_shape_info, dx_shape_info, stride_shape_info, data_flow
    )
    return kernel_name


def _gen_trans_data_case(
    x_dtype,
    dedy_dtype,
    dw_dtype,
    x_shape,
    dedy_shape,
    dw_shape,
    x_format,
    dedy_format,
    dw_format,
    filter_size,
    stride,
    padding,
    dilations=(1, 1, 1, 1),
    groups=1,
    expect="success",
    data_flow="default",
):

    kernel_name = _gen_kernel_name(x_shape, dedy_shape, dw_shape, stride, data_flow)

    fm_list = {
        "shape": util.shape_4d_to_5hd(x_shape, x_dtype, x_format),
        "dtype": x_dtype,
        "format": x_format,
        "ori_shape": x_shape,
        "ori_format": x_format,
    }

    dedy_list = {
        "shape": dedy_shape,
        "dtype": dedy_dtype,
        "format": dedy_format,
        "ori_shape": dedy_shape,
        "ori_format": dedy_format,
    }

    dw_list = {
        "shape": util.shape_4d_to_5hd(dw_shape, dw_dtype, dw_format),
        "dtype": dw_dtype,
        "format": dw_format,
        "ori_shape": dw_shape,
        "ori_format": dw_format,
    }

    filter_sizes = filter_size
    strides = stride
    padding_size = util.gen_padding_size(
        x_shape, dedy_shape, dw_shape, padding, stride, dilations
    )
    data_format = x_format

    if DEBUG_MODE:
        print(
            kernel_name,
            [
                fm_list,
                dedy_list,
                dw_list,
                filter_sizes,
                strides,
                padding,
                dilations,
                groups,
                data_format,
            ],
        )

    return {
        "params": [
            fm_list,
            dedy_list,
            dw_list,
            filter_sizes,
            strides,
            padding_size,
            dilations,
            groups,
            data_format,
        ],
        "case_name": kernel_name,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


def _test_op_check_supported(test_arg):
    from impl.conv2d_backprop_filter_d import check_supported
    out_backprop = {"ori_shape": (1, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
    y = {"ori_shape": (32, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW"}
    x = {"ori_shape": (1, 16, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    filter_size = (32, 16, 1, 1)
    check_supported(x,
                    out_backprop,
                    y,
                    filter_size, (1, 1, 2, 2), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NCHW",
                    kernel_name="conv2d_backprop_filter")


def _gen_conv2d_bp_filter_check_support_case():
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_op_check_supported)


def _test_get_op_support_info(test_arg):
    op_info_testcases = conv2d_bp_filter_ut_testcase.op_support_info_testcase
    for testcase in op_info_testcases:
        formatted_case = _gen_trans_data_case(*testcase)
        params = formatted_case["params"]
        params[0]["format"] = "NC1HWC0"
        get_op_support_info(*params)


def _gen_conv2d_bp_filter_op_case():
    for test_case in conv2d_bp_filter_ut_testcase.conv2d_bp_filter_op_testcase:
        ut_case.add_case(["Ascend910A"], _gen_trans_data_case(*test_case))
    ut_case.add_cust_test_func(test_func=_test_get_op_support_info)


_gen_conv2d_bp_filter_op_case()
_gen_conv2d_bp_filter_check_support_case()




if __name__ == "__main__":
    ut_case.run("Ascend910A")
    sys.exit(0)
