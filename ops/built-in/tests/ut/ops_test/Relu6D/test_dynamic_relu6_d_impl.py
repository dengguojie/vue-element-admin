#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_relu6_d
'''
from op_test_frame.ut import OpUT
import te

ut_case = OpUT("Relu6d", "impl.dynamic.relu6_d", "relu6_d")

case1 = {"params": [{"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (1, -1), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    1.0],
         "case_name": "relu6_d_float16",
         "expect": "success"}

ut_case.add_case("all", case1)


if __name__ == "__main__":
    with te.op.dynamic():
        ut_case.run("Ascend910A")
