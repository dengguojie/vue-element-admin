# # -*- coding:utf-8 -*-
from typing import cast
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("Dynamic_lstm", "impl.dynamic_lstm", "dynamic_lstm")

case1 = {
    'params': [
        {
            'dtype': 'float32',
            'format': 'FRACTAL_NZ',
            'ori_format': 'ND',
            'ori_shape': (1, 1, 512),
            'shape': (1, 32, 1, 16, 16)
        },
        {
            'dtype': 'float32',
            'format': 'FRACTAL_ZN_LSTM',
            'ori_format': 'ND',
            'ori_shape': (768, 1024),
            'shape': (48, 64, 16, 16)
        },
        {
            'dtype': 'float32',
            'format': 'ND',
            'ori_format': 'ND',
            'ori_shape': (1024,),
            'shape': (1024,)
        },
        {
            'dtype': 'float32',
            'format': 'FRACTAL_NZ',
            'ori_format': 'NCHW',
            'ori_shape': (1, 1, 256),
            'shape': (1, 1, 256)
        },
    ],
    'case_name': 'dynamic_lstm_case0',
    'expect': RuntimeError,
    'format_expect': [],
    'support_expect': True,
}
case2 = {
    'params': [
        {
            'dtype': 'float32',
            'format': 'FRACTAL_NZ',
            'ori_format': 'ND',
            'ori_shape': (1, 28, 2, 16, 16),
            'shape': (1, 28, 2, 16, 16)
        },
        {
            'dtype': 'float32',
            'format': 'FRACTAL_ZN_LSTM',
            'ori_format': 'ND',
            'ori_shape': (32, 16, 16, 16),
            'shape': (32, 16, 16, 16)
        },
        {
            'dtype': 'float32',
            'format': 'ND',
            'ori_format': 'ND',
            'ori_shape': (241,),
            'shape': (241,),
        },
        {
            'dtype': 'float32',
            'format': 'FRACTAL_NZ',
            'ori_format': 'NCHW',
            'ori_shape': (1, 4, 2),
            'shape': (1, 4, 2),
        },
    ],
    'case_name': 'dynamic_lstm_case1',
    'expect': RuntimeError,
    'format_expect': [],
    'support_expect': True,
}

ut_case.add_case('Ascend910A', case1)
ut_case.add_case('Ascend910A', case2)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
