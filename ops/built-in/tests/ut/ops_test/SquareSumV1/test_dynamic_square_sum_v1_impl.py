#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_apply_add_sign_d
'''
from op_test_frame.ut import OpUT
import te

ut_case = OpUT("SquareSumV1",
               "impl.dynamic.square_sum_v1",
               "square_sum_v1")

case1 = {"params": [{"shape": (1, -1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 2, 4), "ori_format": "ND", "range": [(1, 1), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND",
                     "ori_shape": (4,), "ori_format": "ND", "range": [(1, None)]},
                    [0, 1], None],
         "case_name": "dynamic_square_sum_v1_001",
         "expect": "success"}

case2 = {"params": [{"shape": (-1, -1, -1), "dtype": "float32", "format": "ND",
                     "ori_shape": (16, 16, 16), "ori_format": "ND", "range": [(1, 16), (1, None), (1, None)]},
                    {"shape": (16,), "dtype": "float32", "format": "ND",
                     "ori_shape": (16,), "ori_format": "ND", "range": [(1, 16)]},
                    [1, 2], True],
         "case_name": "dynamic_square_sum_v1_002",
         "expect": "success"}

ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)

if __name__ == "__main__":
    with te.op.dynamic():
        ut_case.run(["Ascend910A", "Ascend310"])
