#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BroadcastToD", None, None)

case1 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    (1, )],
         "case_name": "broadcast_to_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 3), "dtype": "float32", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float32", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    (3, 3)],
         "case_name": "broadcast_to_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend310")
    exit(0)