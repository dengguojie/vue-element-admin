#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("BnTrainingUpdateV2", None, None)

case1 = {"params": [{"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    0.0001],
         "case_name": "BNTrainingUpdateV2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)