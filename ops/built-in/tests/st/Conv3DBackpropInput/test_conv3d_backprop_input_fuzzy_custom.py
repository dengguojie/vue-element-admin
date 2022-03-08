#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.conv3d_backprop_input import conv3d_backprop_input_generalization
import tbe
import tbe.common.context.op_info as operator_info


def test_conv3d_backprop_input_fuzz_build_static():
    input_list = [
        {'ori_shape': (5,),
         'ori_format': 'ND',
         'dtype': 'int32'},
        {'ori_shape': (1, 1, 1, 256, 64),
         'ori_format': 'DHWCN',
         'dtype': 'float16'},
        {'ori_shape': (1, 8, 56, 56, 64),
         'ori_format': 'NDHWC',
         'dtype': 'float16'},
        {'ori_shape': (1, 8, 56, 56, 256),
         'ori_format': 'NDHWC',
         'dtype': 'float16'}, (1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC',
        'test_conv3d_backprop_input_generalization_static_mode_general_case', {"mode": "keep_rank"}
    ]
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("Conv3DBackpropInput", "Conv3DBackpropInput")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        conv3d_backprop_input_generalization(*input_list)

def test_conv3d_backprop_input_fuzz_build_dynamic():
    input_list = [
        {'ori_shape': (5,),
         'ori_format': 'ND',
         'dtype': 'int32'},
        {'ori_shape': (1, 1, 1, 256, 64),
         'ori_format': 'DHWCN',
         'dtype': 'float16'},
        {'ori_shape': (-1, -1, -1, -1, 64),
         'ori_format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(1, 10), (1, 8), (1, 64), (1, 64), (64, 64)]},
        {'ori_shape': (-1, -1, -1, -1, 256),
         'ori_format': 'NDHWC',
         'dtype': 'float16',
         'ori_range': [(1, 10), (1, 8), (1, 64), (1, 64), (64, 64)]},
        (1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC',
        'test_conv3d_backprop_input_generalization_dynamic_mode_general_case', {"mode": "keep_rank"}
    ]
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("Conv3DBackpropInput", "Conv3DBackpropInput")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        conv3d_backprop_input_generalization(*input_list)


if __name__ == '__main__':
    test_conv3d_backprop_input_fuzz_build_static()
    test_conv3d_backprop_input_fuzz_build_dynamic()