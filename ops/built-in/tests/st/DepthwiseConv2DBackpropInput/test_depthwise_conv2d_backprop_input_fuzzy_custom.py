#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.depthwise_conv2d_backprop_input import depthwise_conv2d_backprop_input_generalization
import tbe
import tbe.common.context.op_info as operator_info


def test_depthwise_conv2d_backprop_input_fuzz_build_static():
    input_list = [
        {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
        {'ori_shape': (2, 12, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
        {'shape': (1, 1, 8, 2000, 16), 'ori_shape': (1, 8, 2000, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
        {'shape': (1, 1, 8, 2000, 16), 'ori_shape': (1, 8, 2000, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
                (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC',
        'test_depthwise_conv2d_backprop_input_generalization_static_mode_general_case', {"mode": "keep_rank"}
    ]
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("DepthwiseConv2dBackpropInput", "DepthwiseConv2dBackpropInput")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        depthwise_conv2d_backprop_input_generalization(*input_list)

def test_depthwise_conv2d_backprop_input_fuzz_build_dynamic():
    input_list = [
        {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (3, 3, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1, 1),(64, 128), (64, 128))},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1, 1),(64, 128), (64, 128))},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_input_generalization_dynamic_mode_general_case', {"mode": "keep_rank"}
    ]
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("DepthwiseConv2dBackpropInput", "DepthwiseConv2dBackpropInput")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        depthwise_conv2d_backprop_input_generalization(*input_list)


if __name__ == '__main__':
    test_depthwise_conv2d_backprop_input_fuzz_build_static()
    test_depthwise_conv2d_backprop_input_fuzz_build_dynamic()