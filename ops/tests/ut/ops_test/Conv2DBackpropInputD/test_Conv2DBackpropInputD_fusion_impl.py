#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the Conv2DBackpropInputD test
"""
import sys

import conv2d_bp_input_ut_testcase
import te.lang.cce as tbe
import util_for_conv2d_bp_input as util
from op_test_frame.ut import OpUT
from te import tvm

ut_case = OpUT(
    "Conv2DBackpropInputD", "impl.conv2d_backprop_input_d", "conv2d_backprop_input_d"
)


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
        dedx = tbe.conv2d_backprop_input_compute(
            filters=w_tensor,
            out_backprop=dedy_tensor,
            filter_sizes=w_shape,
            input_sizes=input_size,
            strides=(stride[2], stride[3]),
            padding=padding,
            dilations=dilations,
            res_dtype=dx_dtype,
            kernel_name=_gen_kernel_name(
                dedy_shape, w_shape, dx_shape, stride, "fusion"
            ),
            group_dict=group_dict,
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


def _gen_conv2d_bp_input_op_fusion_case():
    for fusion_case in conv2d_bp_input_ut_testcase.conv2d_bp_input_fusion_testcase:
        ut_case.add_cust_test_func(
            "Ascend910", test_func=_test_conv2d_bp_input_fusion(*fusion_case)
        )


_gen_conv2d_bp_input_op_fusion_case()


if __name__ == "__main__":
    ut_case.run(["Ascend910", "Ascend710", "Ascend310"])
    sys.exit(0)
