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

InplaceIndexAdd ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("InplaceIndexAdd", None, None)


case1 = {"params": [{"shape": (16, 8, 375), "dtype": "int32", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND"},
                    {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5,), "ori_format": "ND"},
                    {"shape": (16, 5, 375), "dtype": "int32", "format": "ND", "ori_shape": (16, 5, 375), "ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "int32", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND"},
                    1,
                    ],
         "case_name": "inplace_index_add_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (3, 3, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 3, 2), "ori_format": "ND"},
                    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND"},
                    {"shape": (3, 3, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 3, 2), "ori_format": "ND"},
                    {"shape": (3, 3, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 3, 2), "ori_format": "ND"},
                    1,
                    ],
         "case_name": "inplace_index_add_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (8, 5, 128), "dtype": "int32", "format": "ND", "ori_shape": (8, 5, 128), "ori_format": "ND"},
                    {"shape": (4,), "dtype": "int32", "format": "ND", "ori_shape": (4,), "ori_format": "ND"},
                    {"shape": (8, 4, 128), "dtype": "int32", "format": "ND", "ori_shape": (8, 4, 128), "ori_format": "ND"},
                    {"shape": (8, 5, 128), "dtype": "int32", "format": "ND", "ori_shape": (8,5, 128), "ori_format": "ND"},
                    1,
                    ],
         "case_name": "inplace_index_add_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)
