#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from sch_test_frame.ut import OpUT
import tbe
from tbe.common.context import op_context

ut_case = OpUT("conv2d_classify", "conv2d.test_dynamic_conv2d_classify_impl")


def test_binary_conv2d_classify_case0(test_arg):
    try:
        with op_context.OpContext("dynamic"):
            ins = ([{'L1_addr_offset': 0, 'L1_fusion_type': -1, 'L1_workspace_size': -1, 'addr_type': 0, 'data_type': 'float16', 'format': 'NCHW', 'name': 'PlcaceHolder3', 'ori_format': 'NCHW',
                    'ori_range': [[1, -1], [1, -1], [1, -1], [1, -1]], 'ori_shape': [-1, -1, -1, -1], 'range': [[1, -1], [1, -1], [1, -1], [1, -1]], 'sgt_slice_shape': [], 'shape': [-1, -1, -1, -1],
                    'slice_offset': [], 'split_index': 0, 'sub_format': 0, 'total_shape': [-1, -1, -1, -1], 'valid_shape': []}, {'L1_addr_offset': 0, 'L1_fusion_type': -1, 'L1_workspace_size': -1, 'addr_type': 0,
                    'data_type': 'float16', 'format': 'FRACTAL_Z', 'name': 'PlcaceHolder4', 'ori_format': 'NCHW', 'ori_range': [[128, 128], [1280, 1280], [3, 3], [3, 3]], 'ori_shape': [128, 1280, 3, 3],
                    'range': [[720, 720], [8, 8], [16, 16], [16, 16]], 'sgt_slice_shape': [], 'shape': [720, 8, 16, 16], 'slice_offset': [], 'split_index': 0, 'sub_format': 0, 'total_shape': [720, 8, 16, 16],
                    'valid_shape': [], 'input_pattern': 'cube', 'input_op_type': 'Conv2D'}, {'data_type': 0, 'name': 'Conv0', 'shape': 'NULL', 'input_pattern': 'cube', 'input_op_type': 'Conv2D'},
                    {'data_type': 0, 'name': 'Conv1', 'shape': 'NULL', 'input_pattern': 'cube', 'input_op_type': 'Conv2D'}],[{'val': ['NCHW', 'NC1HWC0', 1], 'name': 'trans_TransData_0'}, {'val': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1, 'NCHW', 0], 'name': 'Conv_0'}, {'val': ['NC1HWC0', 'NCHW', 1],
                    'name': 'trans_TransData_3'}],[{'name': 'Conv_0', 'options': {'invalid_data_rm': True}}])
            mode = "Convolution"
            extra_params = None
            tbe.dsl.classify(ins, mode, extra_params)
    except (RuntimeError):
        return False
    else:
        return True


print("test test_binary_conv2d_classify_case0")
ut_case.add_cust_test_func(test_func=test_binary_conv2d_classify_case0)

if __name__ == "__main__":
    ut_case.run(["Ascend910"])
    exit(0)