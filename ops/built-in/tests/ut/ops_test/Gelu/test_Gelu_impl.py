#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("Gelu", "impl.gelu", "gelu")

print('run case 1')
ut_case.add_case(
    'Ascend910A',
    {
        'params': [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                   {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_precision'},
        'case_name': 'gelu_case1',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True,
    },
)

print('run case 2')
ut_case.add_case(
    'Ascend910A',
    {
        'params': [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                   {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_performance'},
        'case_name': 'gelu_case2',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True,
    },
)


print('run case 3')
ut_case.add_case(
    'Ascend910A',
    {
        'params': [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND"},
                   {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_precision'},
        'case_name': 'gelu_case3',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True,
    },
)


print('run case 4')
ut_case.add_case(
    'Ascend910A',
    {
        'params': [{"shape": (16, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND"},
                   {"shape": (16, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_performance'},
        'case_name': 'gelu_case4',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True,
    },
)



if __name__ == '__main__':
    ut_case.run("Ascend910A")
