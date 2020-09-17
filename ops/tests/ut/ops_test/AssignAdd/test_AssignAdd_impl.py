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

AssignAdd ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("AssignAdd", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},  # x
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},  # h
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    ],
         "case_name": "AssignAdd_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (3, 1), "ori_format": "ND"},  # x
                    {"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (3, 1), "ori_format": "ND"},  # h
                    {"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (5, 8, 16, 16), "ori_format": "ND"},
                    ],
         "case_name": "AssignAdd_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (278982,), "dtype": "int8", "format": "ND", "ori_shape": (278982,), "ori_format": "ND"},
                    # x
                    {"shape": (278982,), "dtype": "int8", "format": "ND", "ori_shape": (278982,), "ori_format": "ND"},
                    # h
                    {"shape": (278982,), "dtype": "int8", "format": "ND", "ori_shape": (278982,), "ori_format": "ND"},
                    ],
         "case_name": "AssignAdd_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case2)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case3)

if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310", "Ascend710"])
    exit(0)
