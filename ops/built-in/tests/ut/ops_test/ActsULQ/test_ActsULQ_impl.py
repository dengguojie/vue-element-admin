#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT('acts_ulq', None, None)

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        True,
        8],
        'expect': 'success',
        'case_name': 'test_acts_ulq_fp16_fixed_true'})

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        True,
        8],
        'expect': 'success',
        'case_name': 'test_acts_ulq_fp32_fixed_true'})

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        False,
        8],
        'expect': 'success',
        'case_name': 'test_acts_ulq_fp16_fixed_false'})

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (1,1,1,1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND', 'ori_shape': (1,1,1,1),
         'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        False,
        8],
        'expect': 'success',
        'case_name': 'test_acts_ulq_fp32_fixed_false'})    

if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
