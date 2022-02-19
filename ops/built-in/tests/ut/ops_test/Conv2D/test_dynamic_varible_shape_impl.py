#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import traceback
from op_test_frame.ut import OpUT
import te
import tbe
from te import tvm
from impl.util.platform_adapter import operation
from tbe.common.utils import shape_util

ut_case = OpUT("Conv2D", "impl.dynamic.conv2d", "conv2d")

def cube_varible_shape(test_arg):
    try:
        input_dict_list = [{'L1_addr_offset':0, 'L1_fusion_type':-1, 'L1_workspace_size':-1, 'addr_type':0, 'data_type':'float16',
            'format':'NC1HWC0', 'name':'-1_0_Relu_18_0', 'ori_format':'NCHW', 'ori_range':[[16, 32], [128, 128], [42, 56], [42, 56]],
            'ori_shape':[-1, 128, -1, -1], 'range':[[16, 32], [8, 8], [1764, 3136], [16, 16]], 'sgt_slice_shape':[], 'shape':[-1, 8, -1, 16],
            'slice_offset':[], 'split_index':0, 'sub_format':0, 'total_shape':[-1, 8, -1, -1, 16], 'valid_shape':[]}]
        shape_util._cube_variable_shape(input_dict_list)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding cube_varible_shape")
ut_case.add_cust_test_func(test_func=cube_varible_shape)

def cube_varible_shape1(test_arg):
    try:
        input_dict_list = [{'L1_addr_offset':0, 'L1_fusion_type':-1, 'L1_workspace_size':-1, 'addr_type':0, 'data_type':'float16',
            'format':'NC1HWC0', 'name':'-1_0_Relu_18_0', 'ori_format':'NCHW', 'ori_range':[[16, 16], [128, 128], [42, 56], [42, 56]],
            'ori_shape':[16, 128, -1, -1], 'range':[[16, 16], [8, 8], [42, 56], [42, 56], [16, 16]], 'sgt_slice_shape':[], 'shape':[16, 8, -1, -1, 16],
            'slice_offset':[], 'split_index':0, 'sub_format':0, 'total_shape':[16, 8, -1, -1, 16], 'valid_shape':[], "input_pattern": "cube"},
            {'L1_addr_offset':0, 'L1_fusion_type':-1, 'L1_workspace_size':-1, 'addr_type':0, 'data_type':'float16',
            'format':'FRACTAL_Z', 'name':'-1_0_Relu_18_0', 'ori_format':'NCHW', 'ori_range':[[128, 128], [128, 128], [3, 3], [3, 3]],
            'ori_shape':[128, 128, 3, 3], 'range':[[72, 72], [8, 8], [16, 16], [16, 16]], 'sgt_slice_shape':[], 'shape':[72, 8, 16, 16],
            'slice_offset':[], 'split_index':0, 'sub_format':0, 'total_shape':[72, 8, 16, 16], 'valid_shape':[], "input_pattern": "cube"}]
        shape_util._cube_variable_shape(input_dict_list)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding cube_varible_shape")
ut_case.add_cust_test_func(test_func=cube_varible_shape1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])