#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ConcatV2D", None, None)

case1 = {"params": [[{"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"}],
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    0],
         "case_name": "ConcatV2D_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [[{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0],
         "case_name": "ConcatV2D_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == "__main__":
    # ut_case.run()
    ut_case.run("Ascend910")
    exit(0)