"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dequantize ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("Dequantize",
               "impl.dynamic.dequantize",
               "dequantize")



case1 = {"params": [{"shape": (-1, -1), "dtype": "int8", "format": "ND", "ori_shape": (11, 33),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (11, 33),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    "MIN_COMBINED"],
         "case_name": "dequantize_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1), "dtype": "int8", "format": "ND", "ori_shape": (12, 22),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (12, 22),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    "MIN_FIRST"],
         "case_name": "dequantize_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1), "dtype": "int8", "format": "ND", "ori_shape": (11, 33),
                     "ori_format": "ND", "range": [(1, 10), (1, 10)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (11, 33),
                     "ori_format": "ND", "range": [(1, 10), (1, 10)]},
                    "SCALED"],
         "case_name": "dequantize_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (-2,), "dtype": "int8", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
                    "SCALED"],
         "case_name": "dequantize_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
