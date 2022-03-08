#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

import tbe
from te import platform as cce_conf
from impl.dynamic.power import power
from tbe.common.platform.platform_info import set_current_compile_soc_info

def test_power_01():
    with tbe.common.context.op_context.OpContext("dynamic"):
        power({"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]})

def test_power_02():
    with tbe.common.context.op_context.OpContext("dynamic"):
        power({"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]},
        1.0,
        3.0)

def test_power_03():
    with tbe.common.context.op_context.OpContext("dynamic"):
        power({"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]},
        0.0,
        0.0)

def test_power_04():
    with tbe.common.context.op_context.OpContext("dynamic"):
        power({"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
        "ori_format": "ND", "range": [(1,100)]},
        4.0,
        1.0)

if __name__ == '__main__':
    set_current_compile_soc_info("Ascend710")
    test_power_01()
    test_power_02()
    test_power_03()
    test_power_04()
    exit(0)
