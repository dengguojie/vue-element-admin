"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

RaggedBinCount ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("RaggedBinCount",
               "impl.dynamic.ragged_bin_count", "ragged_bin_count")

case1 = {
    "params": [
        {"shape": (6,), "dtype": "int64", "format": "ND",
         "ori_shape": (6,), "ori_format": "ND"},
        {"shape": (10,), "dtype": "int32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "int32", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (10,), "dtype": "float32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (25,), "dtype": "float32", "format": "ND",
         "ori_shape": (25,), "ori_format": "ND"}
    ],
    "case_name": "RaggedBinCount_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (6,), "dtype": "int64", "format": "ND",
         "ori_shape": (6,), "ori_format": "ND"},
        {"shape": (10,), "dtype": "int64", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "int64", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (10,), "dtype": "float32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (25,), "dtype": "float32", "format": "ND",
         "ori_shape": (25,), "ori_format": "ND"}
    ],
    "case_name": "RaggedBinCount_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {"shape": (-1,), "dtype": "int64", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (10,), "dtype": "int32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "int32", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (10,), "dtype": "float32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "float32", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]}
    ],
    "case_name": "RaggedBinCount_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [
        {"shape": (-1,), "dtype": "int64", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (10,), "dtype": "int64", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "int64", "format": "ND",
         "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (10,), "dtype": "float32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "float32", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]}
    ],
    "case_name": "RaggedBinCount_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [
        {"shape": (-1,), "dtype": "int64", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (10,), "dtype": "int32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "int32", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(30, 80)]},
        {"shape": (10,), "dtype": "float32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "float32", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]}
    ],
    "case_name": "RaggedBinCount_5",
    "expect": "success",
    "support_expect": True
}

case6 = {
    "params": [
        {"shape": (-1,), "dtype": "int64", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]},
        {"shape": (10,), "dtype": "int64", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "int64", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(30, 80)]},
        {"shape": (10,), "dtype": "float32", "format": "ND",
         "ori_shape": (10,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "float32", "format": "ND",
         "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100)]}
    ],
    "case_name": "RaggedBinCount_6",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", ], case1)
ut_case.add_case(["Ascend910A", ], case2)
ut_case.add_case(["Ascend910A", ], case3)
ut_case.add_case(["Ascend910A", ], case4)
ut_case.add_case(["Ascend910A", ], case5)
ut_case.add_case(["Ascend910A", ], case6)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
