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

add ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("Adds")

case1 = {"params": [{"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND"},
                    {"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND"},
                    3.0],
         "case_name": "Adds_1",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case2 = {"params": [{"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND"},
                    {"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND"},
                    3.0],
         "case_name": "Adds_2",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case3 = {"params": [{"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND"},
                    {"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND"},
                    3.0],
         "case_name": "Adds_3",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case4 = {"params": [{"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND"},
                    {"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND"},
                    3.0],
         "case_name": "Adds_4",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
ut_case.add_case(["Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
