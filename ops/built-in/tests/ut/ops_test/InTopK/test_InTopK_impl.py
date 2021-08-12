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
ut_case = OpUT("InTopK", None, None)

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
case11 = {"params": [{"shape": (1216, 1), "dtype": "float32", "ori_shape":(1216, 1), "ori_format":"ND", "format":"ND"},
                    {"shape": (1216, ), "dtype": "int32", "ori_shape":(1216,), "ori_format":"ND", "format":"ND"},
                    {"shape": (1216, 1), "dtype": "float32", "ori_shape":(1216, 1), "ori_format":"ND", "format":"ND"},
                    1],
          "case_name": "in_top_k_11",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case12 = {"params": [{"shape": (2097155, 180), "dtype": "float32", "ori_shape":(2097155, 180), "ori_format":"ND", "format":"ND"},
                    {"shape": (2097155, ), "dtype": "int32", "ori_shape":(2097155,), "ori_format":"ND", "format":"ND"},
                    {"shape": (2097155, 180), "dtype": "float32", "ori_shape":(2097155, 180), "ori_format":"ND", "format":"ND"},
                    1],
          "case_name": "in_top_k_12",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case13 = {"params": [{"shape": (2097155, 1800), "dtype": "float32", "ori_shape":(2097155, 1800), "ori_format":"ND", "format":"ND"},
                    {"shape": (2097155, ), "dtype": "int32", "ori_shape":(2097155,), "ori_format":"ND", "format":"ND"},
                    {"shape": (2097155, 1800), "dtype": "float32", "ori_shape":(2097155, 1800), "ori_format":"ND", "format":"ND"},
                    1],
          "case_name": "in_top_k_13",
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
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case11)
ut_case.add_case(["Ascend910A"], case12)
ut_case.add_case(["Ascend910A"], case13)

def calc_expect_func(predictions, targets, precision, k):
    x = predictions["value"]
    shape = x.shape
    label = targets["value"]

    if k > 0 and k <= shape[1]:
        sub = x - x[list(range(len(label))), label].reshape(-1, 1)
        sub[sub <= 0] = 0
        sub[sub > 0] = 1
        ressum = np.sum(sub, axis=1)
        res=(ressum < k)
    else:
        if k <= 0:
            res = (np.zeros((shape[0], )) > 0)
        else:
            res = (np.zeros((shape[0], )) == 0)
    return res
"""
ut_case.add_precision_case("all", {
    "params": [{"shape": (13, 138), "dtype": "float32", "ori_shape":(13,138), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (13, ), "dtype": "int32", "ori_shape":(13,), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (13, ), "dtype": "uint8", "ori_shape":(13,), "ori_format":"ND", "format":"ND", "param_type": "output"},
               1],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (132, 138), "dtype": "float32", "ori_shape":(13, 138), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (132, ), "dtype": "int32", "ori_shape":(13, ), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (132, ), "dtype": "uint8", "ori_shape":(13, ), "ori_format":"ND", "format":"ND", "param_type": "output"},
               1],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (33, 1024), "dtype": "float32", "ori_shape":(33, 1024), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (33, ), "dtype": "int32", "ori_shape":(33, ), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (33, ), "dtype": "uint8", "ori_shape":(33, ), "ori_format":"ND", "format":"ND", "param_type": "output"},
               1],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (1, 4138), "dtype": "float32", "ori_shape":(1,4138), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (1, ), "dtype": "int32", "ori_shape":(1,), "ori_format":"ND", "format":"ND", "param_type": "input"},
               {"shape": (1, ), "dtype": "uint8", "ori_shape":(1,), "ori_format":"ND", "format":"ND", "param_type": "output"},
               1],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
"""
if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend910A"])
    exit(0)