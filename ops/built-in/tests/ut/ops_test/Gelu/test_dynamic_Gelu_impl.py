#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Gelu", "impl.dynamic.gelu", "gelu")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "Gelu_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND","range":[(1, 100)]},
    ],
    "case_name": "Gelu_2",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [
        {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND","range":[(1, 100), (1, 100)]},
        {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND","range":[(1, 100), (1, 100)]},
    ],
    "case_name": "Gelu_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [
        {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND","range":[(1, 100), (1, 100)]},
        {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND","range":[(1, 100), (1, 100)]},
    ],
    "case_name": "Gelu_4",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)

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
