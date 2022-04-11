#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

import tbe
from te import platform as cce_conf
from impl.dynamic.prod_force_se_a import prod_force_se_a
from tbe.common.platform.platform_info import set_current_compile_soc_info

def test_prod_force_se_a_01():
    with tbe.common.context.op_context.OpContext("dynamic"):
        prod_force_se_a({"shape": (1, 6782976), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 6782976), "ori_format": "ND", "range": ((1, 1), (6782976, 6782976))},
                {"shape": (1, 20348928), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 20348928), "ori_format": "ND", "range": ((1, 1), (20348928, 20348928))},
                {"shape": (1, 1695744), "dtype": "int32", "format": "ND",
                "ori_shape": (1, 1695744), "ori_format": "ND", "range": ((1, 1), (1695744, 1695744))},
                {"shape": (4,), "dtype": "int32", "format": "ND",
                "ori_shape": (4,), "ori_format": "ND", "range": ((4, 4))},
                {"shape": (1, 3, 28328), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 3, 28328), "ori_format": "ND", "range": ((1, 1), (3, 3), (28328, 28328))},
                138, 0, 1, 0, supp_mode="vector")

def test_prod_force_se_a_02():
    input_list = [{"shape": (1, 1656), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 1656), "ori_format": "ND", "range": ((1, 1), (6782976, 6782976))},
                {"shape": (1, 4968), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 4968), "ori_format": "ND", "range": ((1, 1), (20348928, 20348928))},
                {"shape": (1, 414), "dtype": "int32", "format": "ND",
                "ori_shape": (1, 414), "ori_format": "ND", "range": ((1, 1), (1695744, 1695744))},
                {"shape": (4,), "dtype": "int32", "format": "ND",
                "ori_shape": (4,), "ori_format": "ND", "range": ((4, 4))},
                {"shape": (1, 3, 28328), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 3, 28328), "ori_format": "ND", "range": ((1, 1), (3, 3), (28328, 28328))},
                138, 0, 1, 0
                    ]
    with tbe.common.context.op_context.OpContext("dynamic"):
        prod_force_se_a(*input_list)

if __name__ == '__main__':
    set_current_compile_soc_info("Ascend710")
    test_prod_force_se_a_01()
    set_current_compile_soc_info("Ascend910")
    test_prod_force_se_a_02()
    exit(0)
