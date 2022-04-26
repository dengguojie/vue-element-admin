#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the Conv2DBackpropInputD test
"""
import sys
from unittest.mock import MagicMock
from unittest.mock import patch
from op_test_frame.ut import OpUT
import tbe
from tbe import tvm
from tbe.dsl import auto_schedule
from tbe.tvm.target import cce
from te.platform.cce_conf import te_set_version
from impl.add_n import add_n_compute_for_fusion
from impl.conv2d_backprop_input_d import conv2d_backprop_input_d_compute
from impl.conv2d_backprop_input_d import get_op_support_info as dx_get_op_support_info
from impl.fix_pipe import fixpipe_compute
from impl.relu_grad_v2 import relu_grad_v2_compute

import conv2d_bp_input_ut_testcase
import util_for_conv2d_bp_input as util

ut_case = OpUT(
    "Conv2DBackpropInputD", "impl.conv2d_backprop_input_d", "conv2d_backprop_input_d"
)


vals = {("CORE_NUM", ): 48,
    ("CUBE_VECTOR_SPLIT",): True,
    ("UB_SIZE", ): 196608,
    ("L0A_SIZE", ): 65536,
    ("L0B_SIZE", ): 65536,
    ("L1_SIZE", ): 524288,
    ("L0C_SIZE", ): 131072,
    ("SOC_VERSION",): "Ascend910A"
}

support_intrinsic_cube_vector_split = {
    ("Intrinsic_fix_pipe_l0c2ub",) : False,
    ("Intrinsic_fix_pipe_l0c2out",) : True,
    ("Intrinsic_fix_pipe_unit_list",): True,
    ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True,
    ("Intrinsic_data_move_l0c2ub",) : False,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : False,
    ("Intrinsic_mmad", "f162f32",) : True,
    ("Intrinsic_vadd", "float16",) : True,
    ("Intrinsic_vadd", "float32",) : True,
    ("Intrinsic_vadd",) : True,
    ("CUBE_VECTOR_SPLIT",) : True,
}


def side_effects(*args):
    return vals[args]

def check_intrinsic_cube_vector_split(*args, **kwargs):
    return support_intrinsic_cube_vector_split[args]

def _gen_kernel_name(dedy_shape, w_shape, dx_shape, strides, data_flow):
    dedy_shape_info = "_".join([str(i) for i in dedy_shape])
    w_shape_info = "_".join([str(i) for i in w_shape])
    dx_shape_info = "_".join([str(i) for i in dx_shape])
    stride_shape_info = "_".join([str(i) for i in strides])

    kernel_name = "Conv2DBackpropInputD_dy_{}_w_{}_x_{}_stride_{}_data_flow_{}".format(
        dedy_shape_info, w_shape_info, dx_shape_info, stride_shape_info, data_flow
    )
    return kernel_name

    # w_dtype, dedy_dtype, dx_dtype, w_shape, dedy_shape, dx_shape, dedy_format,
    # w_format, dx_format, input_size, stride, padding, dilations, groups, expect, dataflow


def _test_conv2d_bp_input_fusion(
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
    """
    the fusion test for conv2dTranspose
    """

    def _test_conv2d_bp_input_ubfusion_function():
        dedy_tensor = tvm.placeholder(
            util.shape_4d_to_5hd(dedy_shape, dedy_dtype, "NCHW"),
            name="dedy",
            dtype=dedy_dtype,
            attrs={"ori_shape": dedy_shape, "ori_format": "NCHW"},
        )
        w_tensor = tvm.placeholder(
            util.shape_4d_to_fz(w_shape, w_dtype, "NCHW"),
            name="filter",
            dtype=w_dtype,
            attrs={
                "ori_shape": util.get_ori_shape(w_shape, w_dtype, "NCHW"),
                "ori_format": "NCHW",
            },
        )
        dx_5hd = util.shape_4d_to_5hd(dx_shape, dx_dtype, "NCHW")
        dedy_5hd = util.shape_4d_to_5hd(dedy_shape, dedy_dtype, "NCHW")
        group_dict = {
            "groups": 1,
            "g_extend": 1,
            "multiple_extend": 1,
            "dx_c1_extend": dx_5hd[1],
            "dy_c1_extend": dedy_5hd[1],
            "dx_c_ori": dx_shape[1],
            "dedy_c_ori": dedy_shape[1],
            "filter_batch_ori": w_shape[0],
            "filter_c_ori": w_shape[1],
            "filter_ori_format": "NCHW"
        }
        para_dict = {
            "strides": (stride[2], stride[3]),
            "padding": padding,
            "dilations": dilations,
            "res_dtype": dx_dtype,
            "kernel_name": _gen_kernel_name(
                dedy_shape, w_shape, dx_shape, stride, "fusion"
            ),
            "group_dict": group_dict
        }
        dedx = tbe.conv2d_backprop_input_compute(
            filters=w_tensor,
            out_backprop=dedy_tensor,
            filter_sizes=w_shape,
            input_sizes=input_size,
            para_dict=para_dict
        )
        mask_shape = [i.value for i in dedx.shape]
        mask = tvm.placeholder(mask_shape, name="mask", dtype="bool")
        data0 = tvm.const(0, "float16")
        res = tbe.vsel(mask, dedx, data0)
        tensor_list = [w_tensor, dedy_tensor, mask, res]

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": _gen_kernel_name(dedy_shape, w_shape, dx_shape, stride, "fusion"),
            "tensor_list": tensor_list,
        }
        tbe.cce_build_code(sch, config)

    def _test_conv2d_bp_input_ubfusion(test_arg):
        if expect == "success":
            _test_conv2d_bp_input_ubfusion_function()
        elif expect == RuntimeError:
            error_flag = False
            try:
                _test_conv2d_bp_input_ubfusion_function()
            except RuntimeError:
                error_flag = True

            if not error_flag:
                raise RuntimeError(
                    "error_case.: {}".format(
                        _gen_kernel_name(
                            dedy_shape, w_shape, dx_shape, stride, "fusion"
                        )
                    )
                )

    return _test_conv2d_bp_input_ubfusion


def _test_conv2d_bp_input_slice(
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
    """
    the sclice test for conv2d_bp_input
    """
    def _test_conv2d_bp_input_slice_function():
        out_backprop = {"ori_shape": dedy_shape, "dtype": dedy_dtype, "ori_format": dedy_format, "format":"NC1HWC0"}
        filter = {"ori_shape": w_shape, "dtype": w_dtype, "ori_format": w_format, "format":"NC1HWC0"}
        y = {"ori_shape": input_size, "dtype": dedy_dtype, "ori_format": "NCHW", "format":"NC1HWC0"}
        slice_josn = dx_get_op_support_info(
            filter,
            out_backprop,
            y,
            input_size,
            stride,
            padding,
            dilations=(1, 1, 1, 1),
            groups=1,
            data_format=dedy_format,
        kernel_name="conv2d_backprop_input",
        )
        print(f"slice_josn:{slice_josn}")

    def _test_conv2d_bp_input_slice(test_arg):
        if expect == "success":
            _test_conv2d_bp_input_slice_function()
        elif expect == RuntimeError:
            error_flag = False
            try:
                _test_conv2d_bp_input_slice_function()
            except RuntimeError:
                error_flag = True

            if not error_flag:
                raise RuntimeError(
                    "error_case.: {}".format(
                        _gen_kernel_name(
                            dedy_shape, w_shape, dx_shape, stride, "slice"
                        )
                    )
                )


    return _test_conv2d_bp_input_slice


def test_op_check_supported(test_arg):
    from impl.conv2d_backprop_input_d import check_supported
    out_backprop = {"ori_shape": (1, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
    filter = {"ori_shape": (32, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW"}
    y = {"ori_shape": (1, 16, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    input_size = (1, 16, 5, 5)
    check_supported(filter,
                    out_backprop,
                    y,
                    input_size, (1, 1, 2, 2), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NCHW",
                    kernel_name="conv2d_backprop_input")


def test_op_check_supported_dynamic(test_arg):
    from impl.conv2d_backprop_input_d import check_supported
    out_backprop = {"ori_shape": (1, 32, -1, -1), "dtype": "float16", "ori_format": "NCHW"}
    filter = {"ori_shape": (32, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW"}
    y = {"ori_shape": (1, 16, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    input_size = (1, 16, 5, 5)
    check_supported(filter,
                    out_backprop,
                    y,
                    input_size, (1, 1, 2, 2), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NCHW",
                    kernel_name="conv2d_backprop_input")


def test_op_check_supported_error(test_arg):
    from impl.conv2d_backprop_input_d import check_supported
    out_backprop = {"ori_shape": (1, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
    filter = {"ori_shape": (32, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW"}
    y = {"ori_shape": (1, 16, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    input_size = (1, 16, 5, 5)
    try:
        check_supported(filter,
                        out_backprop,
                        y,
                        input_size, (1, 1, 2, 2), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1),
                        groups=0,
                        data_format="NCHW",
                        kernel_name="conv2d_backprop_input")
    except RuntimeError:
        pass

def test_conv2d_bp_input_addn_relugradv2_fusion(test_arg):
    te_set_version("Ascend910")
    with cce():
        out_backprop = tvm.placeholder((28, 4, 32, 95, 16),
                                       name="out_backprop",
                                       attrs={
                                           "format": "NC1HWC0",
                                           "ori_format": "NHWC",
                                           "ori_shape": (28, 32, 95, 62),
                                       },
                                       dtype="float16")
        filters = tvm.placeholder((4, 4, 16, 16),
                                  name="filters",
                                  attrs={
                                      "format": "Fractal_Z",
                                      "ori_format": "NHWC",
                                      "ori_shape": (62, 2, 2, 2)
                                  },
                                  dtype="float16")
        y = {"shape": (28, 1, 96, 96, 16), "ori_shape": (28, 96, 96, 2), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
        input_size_tuple = (28, 96, 96, 2)
        filter_size_tuple = (2, 2, 2, 62)
        strides_tuple = (1, 3, 1, 1)
        # dilation_tuple = (1, 1, 1, 1)
        pads_tuple = (0, 0, 0, 0)
        dx_out = conv2d_backprop_input_d_compute(filters, out_backprop, y, input_size_tuple, strides_tuple, pads_tuple)
        addn_1 = tvm.placeholder(dx_out.shape,
                                 name="addn_1",
                                 attrs={
                                     "format": "NC1HWC0",
                                     "ori_format":"NHWC",
                                     "ori_shape": (28, 96, 96, 2)
                                 },
                                 dtype="float16")
        addn_2 = tvm.placeholder(dx_out.shape,
                                 name="addn_2",
                                 attrs={
                                     "format": "NC1HWC0",
                                     "ori_shape": (28, 96, 96, 2)
                                 },
                                 dtype="float16")
        addn_output = {"shape": dx_out.shape, "ori_shape": (28, 96, 96, 2), "format": "NHWC", "ori_format": "NHWC", "dtype": "float16"}
        addn_result = add_n_compute_for_fusion([dx_out, addn_1, addn_2], addn_output, 3)
        relu_mask = tvm.placeholder((28, 1, 9216, 16),
                                    name="relu_mask",
                                    attrs={
                                        'format': "NC1HWC0",
                                        "ori_shape": (28, 1, 9216, 16)
                                    },
                                    dtype="bool")
        backprops = {"shape": (28, 1, 96, 96, 16), "ori_shape": (28, 96, 96, 2), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
        out = relu_grad_v2_compute(addn_result, relu_mask, backprops)
        tensor_list = [out_backprop, filters, dx_out, addn_1, addn_2, addn_result, relu_mask, out]
        sch = auto_schedule(out)
    te_set_version("Ascend310")

def test_conv2d_bp_input_addn_relugradv2_fusion_opti_schedule(test_arg):
    te_set_version("Ascend910")
    with cce():
        out_backprop = tvm.placeholder((24, 1, 4, 7, 16),
                                       name="out_backprop",
                                       attrs={
                                           "format": "NC1HWC0",
                                           "ori_format": "NHWC",
                                           "ori_shape": (24, 4, 7, 9),
                                       },
                                       dtype="float16")
        filters = tvm.placeholder((1, 1, 16, 16),
                                  name="filters",
                                  attrs={
                                      "format": "Fractal_Z",
                                      "ori_format": "HWCN",
                                      "ori_shape": (1, 1, 4, 9)
                                  },
                                  dtype="float16")
        y = {"shape": (24, 1, 7, 7, 16), "ori_shape": (24, 7, 7, 4), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
        input_size_tuple = (24, 7, 7, 4)
        strides_tuple = (1, 2, 1, 1)
        pads_tuple = (0, 0, 0, 0)
        dx_out = conv2d_backprop_input_d_compute(filters, out_backprop, y, input_size_tuple, strides_tuple, pads_tuple)
        addn_1 = tvm.placeholder(dx_out.shape,
                                 name="addn_1",
                                 attrs={
                                     "format": "NC1HWC0",
                                     "ori_format":"NHWC",
                                     "ori_shape": (24, 7, 7, 4)
                                 },
                                 dtype="float16")
        addn_2 = tvm.placeholder(dx_out.shape,
                                 name="addn_2",
                                 attrs={
                                     "format": "NC1HWC0",
                                     "ori_shape": (24, 7, 7, 4)
                                 },
                                 dtype="float16")
        addn_output = {"shape": dx_out.shape, "ori_shape": (24, 7, 7, 4), "format": "NHWC", "ori_format": "NHWC", "dtype": "float16"}
        addn_result = add_n_compute_for_fusion([dx_out, addn_1, addn_2], addn_output, 3)
        relu_mask = tvm.placeholder((24, 1, 49, 16),
                                    name="relu_mask",
                                    attrs={
                                        'format': "NC1HWC0",
                                        "ori_shape": (24, 1, 49, 16)
                                    },
                                    dtype="bool")
        backprops = {"shape": (24, 1, 7, 7, 16), "ori_shape": (24, 7, 7, 4), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
        out = relu_grad_v2_compute(addn_result, relu_mask, backprops)
        tensor_list = [out_backprop, filters, dx_out, addn_1, addn_2, addn_result, relu_mask, out]
        sch = auto_schedule(out)
    te_set_version("Ascend310")

def test_conv2d_bp_input_addn_relugradv2_fusion_empty_dilate_ub(test_arg):
    te_set_version("Ascend910")
    with cce():
        out_backprop = tvm.placeholder((2, 1, 416, 416, 16),
                                       name="out_backprop",
                                       attrs={
                                           "format": "NC1HWC0",
                                           "ori_format": "NCHW",
                                           "ori_shape": (2, 2, 416, 416),
                                       },
                                       dtype="float16")
        filters = tvm.placeholder((1, 1, 16, 16),
                                  name="filters",
                                  attrs={
                                      "format": "Fractal_Z",
                                      "ori_format": "NCHW",
                                      "ori_shape": (2, 5, 1, 1)
                                  },
                                  dtype="float16")
        y = {"shape": (2, 1, 416, 416, 16), "ori_shape": (2, 5, 416, 416), "format": "NC1HWC0", "ori_format": "NCHW", "dtype": "float16"}
        input_size_tuple = (2, 5, 416, 416)
        strides_tuple = (1, 1, 1, 1)
        pads_tuple = (0, 0, 0, 0)
        dx_out = conv2d_backprop_input_d_compute(filters, out_backprop, y, input_size_tuple, strides_tuple, pads_tuple)
        addn_1 = tvm.placeholder(dx_out.shape,
                                 name="addn_1",
                                 attrs={
                                     "format": "NC1HWC0",
                                     "ori_format":"NHWC",
                                     "ori_shape": (2, 416, 416, 5)
                                 },
                                 dtype="float16")
        addn_2 = tvm.placeholder(dx_out.shape,
                                 name="addn_2",
                                 attrs={
                                     "format": "NC1HWC0",
                                     "ori_shape": (2, 5, 416, 416)
                                 },
                                 dtype="float16")
        addn_output = {"shape": dx_out.shape, "ori_shape": (2, 5, 416, 416), "format": "NCHW", "ori_format": "NHWC", "dtype": "float16"}
        addn_result = add_n_compute_for_fusion([dx_out, addn_1, addn_2], addn_output, 3)
        relu_mask = tvm.placeholder((2, 1, 173056, 16),
                                    name="relu_mask",
                                    attrs={
                                        'format': "NC1HWC0",
                                        "ori_shape": (24, 1, 173056, 16)
                                    },
                                    dtype="bool")
        backprops = {"shape": (2, 1, 416, 416, 16), "ori_shape": (2, 5, 416, 416), "format": "NC1HWC0", "ori_format": "NCHW", "dtype": "float16"}
        out = relu_grad_v2_compute(addn_result, relu_mask, backprops)
        tensor_list = [out_backprop, filters, dx_out, addn_1, addn_2, addn_result, relu_mask, out]
        sch = auto_schedule(out)
    te_set_version("Ascend310")

def _gen_conv2d_bp_input_op_fusion_case():
    for fusion_case in conv2d_bp_input_ut_testcase.conv2d_bp_input_fusion_testcase:
        ut_case.add_cust_test_func(
            "Ascend910A", test_func=_test_conv2d_bp_input_fusion(*fusion_case)
        )


def _gen_conv2d_bp_input_op_slice_case():
    for slice_case in conv2d_bp_input_ut_testcase.conv2d_bp_input_op_slice_testcase:
        ut_case.add_cust_test_func(
            "Ascend910A", test_func=_test_conv2d_bp_input_slice(*slice_case)
        )


def _gen_conv2d_bp_input_check_support_case():
    ut_case.add_cust_test_func("Ascend910A", test_func=test_op_check_supported)
    ut_case.add_cust_test_func("Ascend910A", test_func=test_op_check_supported_dynamic)
    ut_case.add_cust_test_func("Ascend910A", test_func=test_op_check_supported_error)


def test_conv2d_bp_input_addn_relugradv2_fusion_milan(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
            with cce():
                out_backprop = tvm.placeholder((28, 4, 32, 95, 16),
                                               name="out_backprop",
                                               attrs={
                                                   "format": "NC1HWC0",
                                                   "ori_format": "NHWC",
                                                   "ori_shape": (28, 32, 95, 62),
                                               },
                                               dtype="float16")
                filters = tvm.placeholder((4, 4, 16, 16),
                                          name="filters",
                                          attrs={
                                              "format": "Fractal_Z",
                                              "ori_format": "NHWC",
                                              "ori_shape": (62, 2, 2, 2)
                                          },
                                          dtype="float16")
                y = {"shape": (28, 1, 96, 96, 16), "ori_shape": (28, 96, 96, 2), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
                input_size_tuple = (28, 96, 96, 2)
                filter_size_tuple = (2, 2, 2, 62)
                strides_tuple = (1, 3, 1, 1)
                # dilation_tuple = (1, 1, 1, 1)
                pads_tuple = (0, 0, 0, 0)
                dx_out = conv2d_backprop_input_d_compute(filters, out_backprop, y, input_size_tuple, strides_tuple, pads_tuple)
                addn_1 = tvm.placeholder(dx_out.shape,
                                         name="addn_1",
                                         attrs={
                                             "format": "NC1HWC0",
                                             "ori_format":"NHWC",
                                             "ori_shape": (28, 96, 96, 2)
                                         },
                                         dtype="float16")
                addn_2 = tvm.placeholder(dx_out.shape,
                                         name="addn_2",
                                         attrs={
                                             "format": "NC1HWC0",
                                             "ori_shape": (28, 96, 96, 2)
                                         },
                                         dtype="float16")
                addn_output = {"shape": dx_out.shape, "ori_shape": (28, 96, 96, 2), "format": "NHWC", "ori_format": "NHWC", "dtype": "float16"}
                addn_result = add_n_compute_for_fusion([dx_out, addn_1, addn_2], addn_output, 3)
                relu_mask = tvm.placeholder((28, 1, 9216, 16),
                                            name="relu_mask",
                                            attrs={
                                                'format': "NC1HWC0",
                                                "ori_shape": (28, 1, 9216, 16)
                                            },
                                            dtype="bool")
                backprops = {"shape": (28, 1, 96, 96, 16), "ori_shape": (28, 96, 96, 2), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
                out = relu_grad_v2_compute(addn_result, relu_mask, backprops)
                tensor_list = [out_backprop, filters, dx_out, addn_1, addn_2, addn_result, relu_mask, out]
                sch = auto_schedule(out)


def test_conv2d_bp_input_add_milan(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
            with cce():
                out_backprop = tvm.placeholder((28, 4, 32, 95, 16),
                                               name="out_backprop",
                                               attrs={
                                                   "format": "NC1HWC0",
                                                   "ori_format": "NHWC",
                                                   "ori_shape": (28, 32, 95, 62),
                                               },
                                               dtype="float16")
                filters = tvm.placeholder((4, 4, 16, 16),
                                          name="filters",
                                          attrs={
                                              "format": "Fractal_Z",
                                              "ori_format": "NHWC",
                                              "ori_shape": (62, 2, 2, 2)
                                          },
                                          dtype="float16")
                y = {"shape": (28, 1, 96, 96, 16), "ori_shape": (28, 96, 96, 2), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
                input_size_tuple = (28, 96, 96, 2)
                filter_size_tuple = (2, 2, 2, 62)
                strides_tuple = (1, 3, 1, 1)
                # dilation_tuple = (1, 1, 1, 1)
                pads_tuple = (0, 0, 0, 0)
                dx_out = conv2d_backprop_input_d_compute(filters, out_backprop, y, input_size_tuple, strides_tuple, pads_tuple)
                addn_1 = tvm.placeholder(dx_out.shape,
                                         name="addn_1",
                                         attrs={
                                             "format": "NC1HWC0",
                                             "ori_format":"NHWC",
                                             "ori_shape": (28, 96, 96, 2)
                                         },
                                         dtype="float16")
                addn_output = {"shape": dx_out.shape, "ori_shape": (28, 96, 96, 2), "format": "NHWC", "ori_format": "NHWC", "dtype": "float16"}
                out = add_n_compute_for_fusion([dx_out, addn_1], addn_output, 3)
                tensor_list = [out_backprop, filters, dx_out, addn_1, out]
                sch = auto_schedule(out)


def test_conv2d_bp_input_fixpipe_milan(test_arg):
    with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    out_backprop = tvm.placeholder((28, 4, 32, 95, 16),
                                                   name="out_backprop",
                                                   attrs={
                                                       "format": "NC1HWC0",
                                                       "ori_format": "NHWC",
                                                       "ori_shape": (28, 32, 95, 62),
                                                   },
                                                   dtype="float16")
                    filters = tvm.placeholder((4, 4, 16, 16),
                                              name="filters",
                                              attrs={
                                                  "format": "Fractal_Z",
                                                  "ori_format": "NHWC",
                                                  "ori_shape": (62, 2, 2, 2)
                                              },
                                              dtype="float16")
                    y = {"shape": (28, 1, 96, 96, 16), "ori_shape": (28, 96, 96, 2), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
                    input_size_tuple = (28, 96, 96, 2)
                    filter_size_tuple = (2, 2, 2, 62)
                    strides_tuple = (1, 3, 1, 1)
                    # dilation_tuple = (1, 1, 1, 1)
                    pads_tuple = (0, 0, 0, 0)
                    dx_out = conv2d_backprop_input_d_compute(filters, out_backprop, y, input_size_tuple, strides_tuple, pads_tuple)
                    out = fixpipe_compute(dx_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
                    tensor_list = [out_backprop, filters, dx_out, out]
                    sch = auto_schedule(out)

# _gen_conv2d_bp_input_op_fusion_case()
# _gen_conv2d_bp_input_op_slice_case()
# _gen_conv2d_bp_input_check_support_case()
ut_case.add_cust_test_func(test_func=test_conv2d_bp_input_addn_relugradv2_fusion)
ut_case.add_cust_test_func(test_func=test_conv2d_bp_input_addn_relugradv2_fusion_opti_schedule)
ut_case.add_cust_test_func(test_func=test_conv2d_bp_input_addn_relugradv2_fusion_empty_dilate_ub)
ut_case.add_cust_test_func(test_func=test_conv2d_bp_input_addn_relugradv2_fusion_milan)
ut_case.add_cust_test_func(test_func=test_conv2d_bp_input_add_milan)
ut_case.add_cust_test_func(test_func=test_conv2d_bp_input_fixpipe_milan)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend710", "Ascend310"])
    sys.exit(0)
