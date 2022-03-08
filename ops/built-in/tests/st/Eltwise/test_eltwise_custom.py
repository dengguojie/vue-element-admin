#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

import tbe
from te import platform as cce_conf
from impl.dynamic.eltwise import eltwise

def test_eltwise_01():
    input_list = [
[{"shape": (16, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                     {"shape": (16, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]}],
                    {"shape": (16, 16), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (16, 16),
                     "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                    0
    ]
    with tbe.common.context.op_context.OpContext():
        eltwise(*input_list)

def test_eltwise_02():
    input_list_2 = [
[{"shape": (16, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                     {"shape": (16, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]}],
                    {"shape": (16, 16), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (16, 16),
                     "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                    1
    ]
    with tbe.common.context.op_context.OpContext():
        eltwise(*input_list_2)

def test_eltwise_03():
    input_list_3 = [
[{"shape": (16, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                     {"shape": (16, 16), "dtype": "float16",
                      "format": "NCHW", "ori_shape": (16, 16),
                      "ori_format": "NCHW", "range": [(15, 16), (16, 16)]}],
                    {"shape": (16, 16), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (16, 16),
                     "ori_format": "NCHW", "range": [(15, 16), (16, 16)]},
                    2
    ]
    with tbe.common.context.op_context.OpContext():
        eltwise(*input_list_3)

if __name__ == "__main__":
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Ascend910")
    test_eltwise_01()
    test_eltwise_02()
    test_eltwise_03()
    cce_conf.te_set_version(soc_version)
