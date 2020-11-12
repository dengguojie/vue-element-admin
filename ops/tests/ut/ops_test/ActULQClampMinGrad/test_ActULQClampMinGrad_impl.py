#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT('act_ulq_clamp_min_grad', None, None)


ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'}],
        'expect': 'success',
        'case_name': 'test_act_ulq_clamp_min_grad_float16'})

ut_case.add_case(
    ['Ascend910'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'bool', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'}],
        'expect': 'success',
        'case_name': 'test_act_ulq_clamp_min_grad_float32'})


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
