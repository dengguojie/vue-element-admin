#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the Conv2DBackpropInputD test
"""
import sys
from math import ceil as math_ceil

import conv2d_bp_input_ut_testcase
import util_for_conv2d_bp_input as util
from op_test_frame.ut import OpUT
from te import tvm
from te.platform import cce_conf
from te.tvm.target import cce
from topi.generic import auto_schedule
from te.lang.cce import cce_build_code
from impl.trans_data import trans_data_compute
from impl.conv2d_backprop_input_d import conv2d_backprop_input_d_compute


ut_case = OpUT(
    "Conv2DBackpropInputD", "impl.conv2d_backprop_input_d", "conv2d_backprop_input_d"
)
DEBUG_MODE = False


def _gen_kernel_name(dedy_shape, w_shape, dx_shape, strides, data_flow):
    dedy_shape_info = "_".join([str(i) for i in dedy_shape])
    w_shape_info = "_".join([str(i) for i in w_shape])
    dx_shape_info = "_".join([str(i) for i in dx_shape])
    stride_shape_info = "_".join([str(i) for i in strides])

    kernel_name = "Conv2DBackpropInputD_dy_{}_w_{}_x_{}_stride_{}_data_flow_{}".format(
        dedy_shape_info, w_shape_info, dx_shape_info, stride_shape_info, data_flow
    )
    return kernel_name


def _gen_trans_data_case(
    w_dtype,
    dedy_dtype,
    dx_dtype,
    w_shape,
    dedy_shape,
    dx_shape,
    w_format,
    dedy_format,
    dx_format,
    input_size,
    stride,
    padding,
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
    strides = stride
    padding = util.gen_padding_size(dx_shape, w_shape, padding, stride, dilations)
    data_format = dedy_format

    if DEBUG_MODE:
        print(
            kernel_name,
            [
                filters,
                dedy,
                dedx,
                input_sizes,
                strides,
                padding,
                dilations,
                groups,
                data_format,
            ],
        )

    return {
        "params": [
            filters,
            dedy,
            dedx,
            input_sizes,
            strides,
            padding,
            dilations,
            groups,
            data_format,
        ],
        "case_name": kernel_name,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


def _gen_conv2d_bp_input_op_case():
    for test_case in conv2d_bp_input_ut_testcase.conv2d_bp_input_op_testcase:
        ut_case.add_case(["Ascend910A"], _gen_trans_data_case(*test_case))


_gen_conv2d_bp_input_op_case()

def _test_op_check_supported(test_arg):
    from impl.conv2d_backprop_input_d import check_supported
    filter = {"ori_shape": (32, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
    out_backprop = {"ori_shape": (16, 32, 2, 2), "dtype": "float16", "ori_format": "NCHW"}
    y = {"ori_shape": (16, 32, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    input_size = (16, 32, 5, 5)
    check_supported(filter, out_backprop, y, input_size, (1, 1, 2, 2), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW",
                    kernel_name="conv2d_backprop_input")


def _test_nhwc_in_nhwc_out_case_1(test_arg):
    cce_conf.te_set_version('Ascend920A')
    conv_filter = (160, 1, 1, 160)
    out_backprop = (1, 64, 128, 160)
    input_size = (1, 64, 128, 160)
    strides = (1, 1)
    pads = (0, 0, 0, 0)
    dilations = (1, 1, 1, 1)
    data_type = "float16"
    param_dict = {
        "strides" : strides,
        "padding" : pads,
        "dilations" : dilations,
    }
    with cce():
        fmap_ori = tvm.placeholder(out_backprop, name="fmap_ori", dtype=data_type)
        weight_ori = tvm.placeholder(conv_filter, name="weight_ori", dtype=data_type)
        fmap = trans_data_compute(fmap_ori, None, src_format="NHWC", dst_format="NC1HWC0")
        weight = trans_data_compute(weight_ori, None, src_format = "NHWC", dst_format="FRACTAL_Z")
        y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NHWC", "format" : "NC1HWC0"}
        conv_res = conv2d_backprop_input_d_compute(weight, fmap, y, input_size, strides, pads)
        src_n, src_c1, src_hw, src_c0 =  tuple(i.value for i in conv_res.shape)
        out = trans_data_compute(conv_res, {"shape" : (src_n, src_hw, src_c1*src_c0)}, src_format="NC1HWC0",
                                   dst_format="NHWC")
        tensor_list = [weight_ori, fmap_ori, out]
        sch = auto_schedule(out)
        config = {
            "print_ir" : False,
            "need_build" : True,
            "name" : "conv2d_bp_input_ut_testcase_1",
            "tensor_list" : tensor_list
        }
        cce_build_code(sch, config)
    cce_conf.te_set_version('Ascend910A')


ut_case.add_cust_test_func(test_func=_test_nhwc_in_nhwc_out_case_1)

def _gen_conv2d_bp_input_check_support_case():
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_op_check_supported)

_gen_conv2d_bp_input_check_support_case()


if __name__ == "__main__":
    ut_case.run("Ascend910A")
    sys.exit(0)
