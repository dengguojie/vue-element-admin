#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("Tanh", "impl.tanh", "tanh")

ut_case.add_case(
    'Ascend910A',
    {
        'params': [{"shape": (1, 3, 100, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 100, 16), "ori_format": "ND"},
                   {"shape": (1, 3, 100, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 3, 100, 16), "ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_performance'},
        'case_name': 'tanh_case1',
        'expect': 'success',
        'support_expect': True,
    },
)

ut_case.add_case(
    'Ascend910A',
    {
        'params': [{"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16), "ori_format": "ND"},
                   {"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16), "ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_performance'},
        'case_name': 'tanh_case1',
        'expect': 'success',
        'support_expect': True,
    },
)


if __name__ == '__main__':
    ut_case.run("Ascend910A")