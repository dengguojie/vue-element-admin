#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import te.lang.cce as tbe
from te import tvm
from op_test_frame.ut import OpUT
from impl.depthwise_conv2d_backprop_filter_d import depthwise_conv2d_backprop_filter_d_compute


ut_case = OpUT("DepthwiseConv2dBackpropFilterD", "impl.depthwise_conv2d_backprop_filter_d",
               "depthwise_conv2d_backprop_filter_d")

conv2d_bp_input_fusion_testcase = [
    # fmap, dedy, filter_grads, filter, strides, dilations, pads, data_format, expect
    ((2, 16, 28, 28), (2, 16, 26, 26), (3, 3, 16, 1), (3, 3, 16, 1), (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NCHW", "success"),
    ((2, 28, 28, 16), (2, 26, 26, 16), (3, 3, 16, 1), (3, 3, 16, 1), (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NHWC", "success"),
    ((2, 16, 28, 28), (2, 16, 26, 26), (3, 3, 16, 1), (3, 3, 16, 1), (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0), "NCHW", RuntimeError),
    ((2, 16, 28, 28), (2, 16, 26, 26), (3, 3, 16, 2), (3, 3, 16, 1), (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NCHW", RuntimeError),
    ((2, 16, 28, 28), (10, 16, 26, 26), (3, 3, 16, 1), (3, 3, 16, 1), (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NCHW", RuntimeError),
]

def _shape_ND_to_NC1HWC0(shape, data_format, dtype="float16"):
    if data_format.upper() == "NCHW":
        n, c, h, w = shape
    else:  # NCHW
        n, h, w, c = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (n, c1, h, w, c0)


def _test_depthwise_conv2d_bp_filter_trans_data_fusion(
    fmap_shape,
    dedy_shape,
    filter_grad,
    filter_size,
    strides=(1, 1, 1, 1),
    dilations=(1, 1, 1, 1),
    pads=(0, 0, 0, 0),
    data_format="NCHW",
    expect="success",
):
    """
    the fusion test for depthwise-dw
    """

    def _test_depthwise_conv2d_bp_filter_fusion_function():
        fmap_tensor = tvm.placeholder(
            _shape_ND_to_NC1HWC0(fmap_shape, data_format),
            name="fmap",
            dtype="float16",
            attrs={
                "ori_shape": fmap_shape, 
                "ori_format": data_format,
            },
        )
        dedy_tensor = tvm.placeholder(
            _shape_ND_to_NC1HWC0(dedy_shape, data_format),
            name="dedy",
            dtype="float16",
            attrs={
                "ori_shape": dedy_shape,
                "ori_format": data_format,
            },
        )

        dedw = depthwise_conv2d_backprop_filter_d_compute(
            fmap_tensor,
            dedy_tensor,
            filter_grad={"ori_shape": filter_grad, "ori_format": "HWCK", "dtype": "float32"},
            filter_size=list(filter_size),
            strides=strides,
            dilations=dilations,
            pads=pads,
        )

        tensor_list_input = [fmap_tensor, dedy_tensor]

        with tvm.target.cce():
            sch = tbe.auto_schedule(dedw)
        real_outs = sch.cce_special["real_out_tensor"]
        tensor_list = tensor_list_input + real_outs
        config = {"name": "depthwise_dw_trans_data_fusion", "tensor_list": tensor_list}

        tbe.cce_build_code(sch, config)

    def _test_depthwise_conv2d_bp_filter_fusion(test_arg):
        if expect == "success":
            _test_depthwise_conv2d_bp_filter_fusion_function()
        elif expect == RuntimeError:
            try:
                _test_depthwise_conv2d_bp_filter_fusion_function()
            except RuntimeError:
                print("Depthwise_dw trans_data fusion test mock")

    return _test_depthwise_conv2d_bp_filter_fusion


def _gen_conv2d_bp_filter_op_fusion_case():
    # Add depthwise trans_data fusion cases
    for fusion_case in conv2d_bp_input_fusion_testcase:
        ut_case.add_cust_test_func(
            "Ascend910A", test_func=_test_depthwise_conv2d_bp_filter_trans_data_fusion(*fusion_case)
        )

_gen_conv2d_bp_filter_op_fusion_case()

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
