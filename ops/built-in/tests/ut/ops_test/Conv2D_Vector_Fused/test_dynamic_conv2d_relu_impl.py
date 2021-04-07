#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm
import te.lang.cce as tbe
import te.lang.dynamic as dynamic
import te.lang.base.operation_impl as operation
from impl.dynamic import relu
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv2D", "impl.dynamic.conv2d",
               "conv2d")

def test_conv2d_relu_dynamic(test_arg):
    print("[ conv_relu_dynamic_case ]")
    inputs = {"ori_shape": [1, 32, -1, -1], "dtype": "float16", "ori_format": "NCHW",
              "range": [(1, 1), (32, 32), (10, 25), (10, 25)]}
    weights = {"ori_shape": [64, 32, 1, 1], "dtype": "float16", "ori_format": "NCHW",
              "range": [(64, 64), (32, 32), (1, 1), (1, 1)]}
    bias = None
    offset_w = None
    outputs = {"dtype": "float16"}
    strides = [-1, -1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = (1, 1, 1, 1)
    groups = 1
    data_format = "NCHW"
    offset_x = 0
    kernel_name_val="conv_relu_dynamic"
    relu_outputs={"dtype": "float16"}
    with operation.OperatorContext(operation.OpMode.DYNAMIC) as opc:
        opc.set_op_type("conv_relu")
        with operation.ComputeContext():
            conv = operation.get_fusion_compute('Conv2D')
            relu = operation.get_fusion_compute('Relu')
            conv_res = conv(inputs,
                            weights,
                            bias,
                            offset_w,
                            outputs,
                            strides,
                            pads,
                            dilations,
                            groups=groups,
                            data_format=data_format,
                            offset_x=offset_x,
                            kernel_name="conv2d")

            relu_res = relu(conv_res['op_res'][0], relu_outputs, "relu")
            with tvm.target.cce():
                sch = tbe.auto_schedule(relu_res['op_res'])
            tensor_list = list(conv_res['op_placeholder']) + \
                        list(relu_res['op_res'])
            config = {"name": kernel_name_val,
                "tensor_list": tensor_list,
                "build_args": {"constant_realize_extent_in_infer_bound": False}}

        dynamic.build(sch, config)

# ut_case.add_cust_test_func(test_func=test_conv2d_relu_dynamic)

if __name__ == '__main__':
    ut_case.add_cust_test_func(test_func=test_conv2d_relu_dynamic)
    exit(0)
