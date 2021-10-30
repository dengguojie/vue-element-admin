#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.dynamic.get_shape import get_shape
import tbe
from tbe.common.platform import set_current_compile_soc_info

ut_case = OpUT("GetShape", "impl.dynamic.get_shape", "get_shape")


def test_1951_int64_get_shape_001(test_arg):
    set_current_compile_soc_info('Ascend710')
    with tbe.common.context.op_context.OpContext("dynamic"):
        get_shape([
            {
                "shape": (16, 8),
                "dtype": "int64",
                "format": "ND",
                "ori_shape": (16, 8),
                "ori_format": "ND"
            },
            {
                "shape": (16, 2, 4),
                "dtype": "int64",
                "format": "ND",
                "ori_shape": (16, 2, 4),
                "ori_format": "ND"
            },
            {
                "shape": (4, 4, 4, 4),
                "dtype": "int64",
                "format": "ND",
                "ori_shape": (4, 4, 4, 4),
                "ori_format": "ND"
            },
        ], {
            "shape": (9,),
            "dtype": "int32",
            "format": "ND",
            "ori_shape": (9,),
            "ori_format": "ND"
        }, "test_1951_int64_get_shape_001_tf")
    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(test_func=test_1951_int64_get_shape_001)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run('Ascend710')
