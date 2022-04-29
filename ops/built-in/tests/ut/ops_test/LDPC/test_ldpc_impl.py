"""
Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

LDPC ut case
"""

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("LDPC","impl.ldpc", "ldpc")

case1 = {"params": [{"shape": (10240, 128), "dtype": "float16", "format": "ND", "ori_shape": (10240, 128),
                     "ori_format": "ND"},
                    {"shape": (6144, 6), "dtype": "int32", "format": "ND",
                    "ori_shape": (6144, 6),"ori_format": "ND"},
                    {"shape": (10240, 128), "dtype": "float16", "format": "ND", "ori_shape": (10240, 128),
                     "ori_format": "ND"},
                     {"shape": (128, 1280), "dtype": "uint8", "format": "ND", "ori_shape": (128, 1280),
                     "ori_format": "ND"},
                     {"shape": (6144, 6, 128), "dtype": "float16", "format": "ND", "ori_shape": (6144, 6, 128),
                     "ori_format": "ND"}],
         "case_name": "ldpc_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case_lis = [case1]
for ele in case_lis:
    ut_case.add_case(["Ascend910A"], ele)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
