#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
from te.platform.cce_conf import te_set_version
from impl.combined_non_max_suppression import CNMS


def test_cnms_001():
    '''
    for combined_non_max_suppression single op
    '''
    input_list = [{'shape': [1, 1, 4, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [1, 1, 4, 29782],
                           'ori_format': 'ND'},
                          {'shape': [1, 1, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [1, 1, 29782],
                           'ori_format': 'ND'},
                          [{'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (100,)},
                          {'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (100,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)}], 0.5, 0.5, 100, 100, "test_cnms_001"
                          ]
    obj = CNMS(*input_list)
    obj.cnms_compute()


def test_cnms_002():
    '''
    for combined_non_max_suppression single op
    '''
    input_list = [{'shape': [2, 2, 4, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [2, 2, 4, 29782],
                           'ori_format': 'ND'},
                          {'shape': [2, 2, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [2, 2, 29782],
                           'ori_format': 'ND'},
                          [{'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (100,)},
                          {'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (50,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)}], 0.7, 0.7, 100, 50, "test_cnms_002"
                          ]
    obj = CNMS(*input_list)
    obj.cnms_compute()


def test_cnms_003():
    '''
    for combined_non_max_suppression single op
    '''
    input_list = [{'shape': [2, 50, 4, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [2, 2, 50, 29782],
                           'ori_format': 'ND'},
                          {'shape': [2, 50, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [2, 50, 29782],
                           'ori_format': 'ND'},
                          [{'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (100,)},
                          {'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (50,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)}], 0.7, 0.7, 100, 100, "test_cnms_003"
                          ]
    obj = CNMS(*input_list)
    obj.cnms_compute()

if __name__ == '__main__':
    soc_version = te.platform.cce_conf.get_soc_spec("SOC_VERSION")
    te_set_version("Ascend920A", "VectorCore")
    test_cnms_001()
    test_cnms_002()
    test_cnms_003()
    te_set_version(soc_version)
