#!/usr/bin/env python
from op_test_frame.ut import OpUT


ut_case = OpUT('IFMR', 'impl.dynamic.ifmr', 'ifmr')

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (512,), 'dtype': 'int32', 'format': 'ND', 'ori_shape': (512,), 'ori_format': 'ND', 'range': [(512, 512)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        0.999999,
        0.999999,
        [0.7, 1.3],
        0.01,
        True],
        'expect': 'success',
        'case_name': 'test_ifmr_float16_with_offset',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (512,), 'dtype': 'int32', 'format': 'ND', 'ori_shape': (512,), 'ori_format': 'ND', 'range': [(512, 512)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        0.999999,
        0.999999,
        [0.7, 1.3],
        0.01,
        True],
        'expect': 'success',
        'case_name': 'test_ifmr_float32_with_offset',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (512,), 'dtype': 'int32', 'format': 'ND', 'ori_shape': (512,), 'ori_format': 'ND', 'range': [(512, 512)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        0.999999,
        0.999999,
        [0.7, 1.3],
        0.01,
        False],
        'expect': 'success',
        'case_name': 'test_ifmr_float16_without_offset',
        'support_expect': True})


if __name__ == '__main__':
    ut_case.run('Ascend910A')
