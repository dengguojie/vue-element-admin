#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ConcatOffsetD", None, None)

list0 = {"shape": (4,), "ori_shape": (4,), "format": "ND", "ori_format": "ND", "dtype": "int32"}
case1 = {"params": [[list0, list0, list0, list0, list0],
                    [list0, list0, list0, list0, list0],
                    2],
         "case_name": "ConcatOffsetD_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [[list0, list0, list0, list0, list0],
                    [list0, list0, list0, list0, list0],
                    10],
         "case_name": "ConcatOffsetD_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")