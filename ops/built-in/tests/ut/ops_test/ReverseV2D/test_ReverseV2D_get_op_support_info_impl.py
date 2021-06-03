#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ReverseV2D", "impl.reverse_v2_d", "get_op_support_info")

case1 = {"params": [{"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    [1,2]],
        "case_name": "ReverseV2D_1",
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

if __name__ == "__main__":
    ut_case.run("Ascend910")



