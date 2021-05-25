#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT('wts_arq', None, None)

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [16, 1, 1, 1], "param_type": "required", "shape": [16, 1, 1, 1]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [16, 1, 1, 1], "param_type": "required", "shape": [16, 1, 1, 1]},
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        8,
        False],
        'expect': 'success',
        'case_name': 'test_wts_arq_float32_offset_false'})

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [1, 1, 1, 1], "param_type": "required", "shape": [1, 1, 1, 1]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [1, 1, 1, 1], "param_type": "required", "shape": [1, 1, 1, 1]},
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        8,
        False],
        'expect': 'success',
        'case_name': 'test_wts_arq_float32_per_tensor'})

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [16, 1, 1, 1], "param_type": "required", "shape": [16, 1, 1, 1]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [16, 1, 1, 1], "param_type": "required", "shape": [16, 1, 1, 1]},
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        8,
        True],
        'expect': 'success',
        'case_name': 'test_wts_arq_float32_offset_true'})

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [16, 1, 1, 1], "param_type": "required", "shape": [16, 1, 1, 1]},
        {"dtype": "float32", "format": "NCHW", "ori_format":"NCHW",
         "ori_shape": [16, 1, 1, 1], "param_type": "required", "shape": [16, 1, 1, 1]},
        {'dtype': 'float32', 'format': 'NCHW', 'ori_format': 'NCHW',
         'ori_shape': [16, 6, 5, 5], 'param_type': 'required',
         'shape': [16, 6, 5, 5]},
        6,
        False],
        'expect': ValueError,
        'case_name': 'test_wts_arq_num_bits_invalid'})


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
