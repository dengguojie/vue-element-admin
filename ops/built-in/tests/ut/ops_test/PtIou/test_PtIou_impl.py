#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("PtIou", op_func_name="iou")

case1 = {"params": [{"shape": (1, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 4), "ori_format": "ND"},
                    {"shape": (1, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 4), "ori_format": "ND"},
                    {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"}],
         "case_name": "pt_iou_1",
         "expect": "success"}
ut_case.add_case(["Ascend910A"], case1)


if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
