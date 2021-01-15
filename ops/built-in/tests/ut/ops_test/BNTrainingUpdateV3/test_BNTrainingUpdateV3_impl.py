#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("BnTrainingUpdateV3", None, None)

case1 = {"params": [{"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,1,1,16),"ori_format": "NC1HWC0"},
                    0.0001],
         "case_name": "BNTrainingUpdateV3_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (2,1,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,2,1,1,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,1,2,1,1,16),"ori_format": "NDC1HWC0"},
                    0.0001],
         "case_name": "BNTrainingUpdateV3_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    # ut_case.run()
    exit(0)
