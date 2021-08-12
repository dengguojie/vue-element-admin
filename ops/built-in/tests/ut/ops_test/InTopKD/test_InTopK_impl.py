"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

InTopK ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("InTopKD", "impl.dynamic.in_top_k", "in_top_k")

case1 = {"params": [{"shape": (1, 1), "dtype": "float32", "ori_shape":(1,1), "ori_format":"ND", "format":"ND"},
                    {"shape": (1, ), "dtype": "int32", "ori_shape":(1,), "ori_format":"ND", "format":"ND"},
                    {"shape": (1, 1), "dtype": "float32", "ori_shape":(1,1), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (3, 231), "dtype": "float32", "ori_shape":(3,231), "ori_format":"ND", "format":"ND"},
                    {"shape": (3, ), "dtype": "int32", "ori_shape":(3,), "ori_format":"ND", "format":"ND"},
                    {"shape": (3, 231), "dtype": "float32", "ori_shape":(3,231), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (3, 256), "dtype": "float32", "ori_shape":(3,256), "ori_format":"ND", "format":"ND"},
                    {"shape": (3, ), "dtype": "int32", "ori_shape":(3,), "ori_format":"ND", "format":"ND"},
                    {"shape": (3, 256), "dtype": "float32", "ori_shape":(3,256), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 63), "dtype": "float32", "ori_shape":(32,63), "ori_format":"ND", "format":"ND"},
                    {"shape": (32, ), "dtype": "int32", "ori_shape":(32,), "ori_format":"ND", "format":"ND"},
                    {"shape": (32, 63), "dtype": "float32", "ori_shape":(32,63), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (13, 138), "dtype": "float32", "ori_shape":(13,138), "ori_format":"ND", "format":"ND"},
                    {"shape": (13, ), "dtype": "int32", "ori_shape":(13,), "ori_format":"ND", "format":"ND"},
                    {"shape": (13, 138), "dtype": "float32", "ori_shape":(13,138), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (87, 18), "dtype": "float32", "ori_shape":(87, 18), "ori_format":"ND", "format":"ND"},
                    {"shape": (87, ), "dtype": "int32", "ori_shape":(87,), "ori_format":"ND", "format":"ND"},
                    {"shape": (87, 18), "dtype": "float32", "ori_shape":(87, 18), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (16, 180), "dtype": "float32", "ori_shape":(16, 180), "ori_format":"ND", "format":"ND"},
                    {"shape": (16, ), "dtype": "int32", "ori_shape":(16,), "ori_format":"ND", "format":"ND"},
                    {"shape": (16, 180), "dtype": "float32", "ori_shape":(16, 180), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (57, 180), "dtype": "float32", "ori_shape":(57, 180), "ori_format":"ND", "format":"ND"},
                    {"shape": (57, ), "dtype": "int32", "ori_shape":(57,), "ori_format":"ND", "format":"ND"},
                    {"shape": (57, 180), "dtype": "float32", "ori_shape":(57, 180), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (57, 1800), "dtype": "float32", "ori_shape":(57, 1800), "ori_format":"ND", "format":"ND"},
                    {"shape": (57, ), "dtype": "int32", "ori_shape":(57,), "ori_format":"ND", "format":"ND"},
                    {"shape": (57, 1800), "dtype": "float32", "ori_shape":(57, 1800), "ori_format":"ND", "format":"ND"},
                    1],
         "case_name": "in_top_k_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (57, 18000), "dtype": "float32", "ori_shape":(57, 18000), "ori_format":"ND", "format":"ND"},
                    {"shape": (57, ), "dtype": "int32", "ori_shape":(57,), "ori_format":"ND", "format":"ND"},
                    {"shape": (57, 18000), "dtype": "float32", "ori_shape":(57, 18000), "ori_format":"ND", "format":"ND"},
                    1],
          "case_name": "in_top_k_10",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

    

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case10)

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run()
        exit(0)