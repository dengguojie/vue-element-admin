#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe
from impl.dynamic.non_zero import non_zero

ut_case = OpUT("NonZero", "impl.dynamic.non_zero", "non_zero")

def non_zero_test_001(test_arg):
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        non_zero({"shape": (1000, 21136), "dtype": "float32", "format": "ND", "ori_shape": (1000, 21136), "ori_format": "ND",
                  "range": [(1000, 1000), (21136, 21136)]},
                 {"shape": (1000, 21136), "dtype": "int32", "format": "ND", "ori_shape": (1000, 21136), "ori_format": "ND",
                  "range": [(1000, 1000), (21136, 21136)]},
                  True,
                  "test_non_zero_001")
    set_current_compile_soc_info(test_arg)

ut_case.add_cust_test_func(test_func=non_zero_test_001)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run('Ascend910A')
