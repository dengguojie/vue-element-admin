#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.dynamic.conv2d", "conv2d")

def test_conv2d_dynamic_transdata_fusion(test_arg):
    from impl.dynamic.conv2d import conv2d_fusion_compute
    from impl.dynamic.trans_data import trans_data_fusion_compute
    from impl.dynamic.conv2d_data_rm import conv2d_data_rm_compute
    from tbe.common.context import op_context
    import tbe.dsl.base.operation as operation
    from tbe.dsl import auto_schedule
    from tbe.dsl import build
    from tbe import tvm

    bias_flag = False
    strides = [-1, -1, -1, -1]
    pads = [1, 1, 1, 1]
    dilations = [1, 1, 1, 1]
    groups = 1
    in_dtype = "float16"
    kernel_name = "conv2d_binary_transdata_fusion"

    outputs = {"dtype": in_dtype, "ori_format": "NCHW"}

    with operation.dynamic():
        with operation.ComputeContext():
            batch_n = operation.var("batch_n", (1, None))
            c_in = operation.var("c_in")
            fmap_h = operation.var("fmap_h", (1, None))
            fmap_w = operation.var("fmap_w", (1, None))
            c_out = operation.var("c_out")
            k_h = operation.var("k_h")
            k_w = operation.var("k_w")

            fmap_shape_nchw = (batch_n, c_in, fmap_h, fmap_w)
            Ci0 = 16
            Co0 = 16
            Ci1 = (c_in + Ci0 - 1)//Ci0
            Co1 = (c_out + Co0 - 1)//Co0

            shape_w_fracz = (k_h*k_w*Ci1, Co1, Co0, Ci0)

            data = tvm.placeholder(fmap_shape_nchw, name='fmap', dtype=in_dtype, attrs={"range": [[1, -1], [1, -1], [1, -1], [1, -1]],
                                                                                        "ori_shape": [-1, -1, -1, -1],
                                                                                        "format": "NCHW",
                                                                                        "ori_format": "NCHW",})
            weight = tvm.placeholder(shape_w_fracz, name='weight', dtype=in_dtype, attrs={'ori_shape': [-1, -1, -1, -1], 'ori_format': "NCHW", "format": "FRACTAL_Z",})
            bias = tvm.placeholder((Co0*Co1,), name='bias', dtype=in_dtype, attrs={'ori_shape': [-1], 'ori_format': "ND", 'format': "ND"}) if bias_flag else None

            conv_in = trans_data_fusion_compute(data, None, "NCHW", "NC1HWC0")

            conv_out = conv2d_fusion_compute(conv_in, weight, bias, None, outputs, strides,
                                             pads, dilations, groups, "NCHW", 0, kernel_name,
                                             options={"invalid_data_rm": True})

            dst = {"ori_shape": [-1, c_out, -1, -1]}
            res = trans_data_fusion_compute(conv_out, dst, "NC1HWC0", "NCHW")
            res = conv2d_data_rm_compute(res)

            tensor_list = [data, weight, bias, res] if bias_flag else [data, weight, res]

            with tvm.target.cce():
                sch = auto_schedule(res)

            config = {"name": kernel_name,
                      "tensor_list": tensor_list,
                      "build_args": {"constant_realize_extent_in_infer_bound": False}}
        # build(sch, config)


print("test Conv2D dynamic transdata fusion running")
ut_case.add_cust_test_func("Ascend910A", test_func=test_conv2d_dynamic_transdata_fusion)
ut_case.run(['Ascend910A'])
