#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from tbe import tvm
import tbe.dsl as tbe
from tbe.dsl.unify_schedule.unify_auto_schedule import build
import tbe.dsl.base.operation as operation
from impl.dynamic.conv2d import _conv2d_compute
from impl.dynamic.bn_training_reduce import bn_training_reduce_compute
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv2D", "impl.dynamic.conv2d",
               "conv2d")

def test_conv2d_bn1_dynamic(test_arg):
    print("[ conv_relu_dynamic_case ]")
    inputs = {"ori_shape": [1, 32, -1, -1], "dtype": "float16", "ori_format": "NCHW",
              "range": [(1, 1), (32, 32), (10, 25), (10, 25)]}
    weights = {"ori_shape": [64, 32, 1, 1], "dtype": "float16", "ori_format": "NCHW",
              "range": [(64, 64), (32, 32), (1, 1), (1, 1)]}
    bias = None
    offset_w = None
    outputs = {"dtype": "float16", "ori_format":"NCHW"}
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = (1, 1, 1, 1)
    groups = 1
    data_format = "NCHW"
    offset_x = 0
    kernel_name_val="conv_bn1_dynamic"
    with operation.dynamic():
        with operation.ComputeContext():
            conv_res = _conv2d_compute(inputs,
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
            conv_out = conv_res['op_res'][0]
            axis = [0, 2, 3]
            bn_out = bn_training_reduce_compute(conv_out, {"format":"NC1HWC0"}, None, axis)

            out = [conv_out, bn_out[0], bn_out[1]]
            with tvm.target.cce():
                sch = tbe.auto_schedule(out)

            tensor_list = list(conv_res['op_placeholder'])
            real_outs = sch[0].addition["real_outs"]
            tensor_list.extend(real_outs)
            
            config = {"name": kernel_name_val,
                "tensor_list": tensor_list,
                "build_args": {"constant_realize_extent_in_infer_bound": False}}

        build(sch, config)

ut_case.add_cust_test_func(test_func=test_conv2d_bn1_dynamic)

if __name__ == '__main__':
    # ut_case.add_cust_test_func(test_func=test_conv2d_bn1_dynamic)
    ut_case.run("Ascend910A")
    exit(0)
