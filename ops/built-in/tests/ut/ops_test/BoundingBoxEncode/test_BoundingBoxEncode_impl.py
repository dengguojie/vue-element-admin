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

BoundingBoxEncode ut case
"""
from unittest.mock import patch
from unittest.mock import MagicMock
from op_test_frame.ut import OpUT
ut_case = OpUT("BoundingBoxEncode", None, None)

case1 = {"params": [{"shape": (15, 32, 1, 4), "dtype": "float16", "format": "ND", "ori_shape": (15, 32, 1, 4),"ori_format": "ND"}, #x
                    {"shape": (15, 32, 1, 4), "dtype": "float16", "format": "ND", "ori_shape": (15, 32, 1, 4),"ori_format": "ND"}, #h
                    {"shape": (15, 32, 1, 4), "dtype": "float16", "format": "ND", "ori_shape": (15, 32, 1, 4),"ori_format": "ND"},
                    (1.0, 2.1, 0.0, 1.8),(2.1, 1.0, 3.6, 1.9)],
         "case_name": "BoundingBoxEncode_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (5, 13, 64, 16, 4), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 64, 16, 4),"ori_format": "ND"}, #x
                    {"shape": (5, 13, 64, 16, 4), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 64, 16, 4),"ori_format": "ND"}, #h
                    {"shape": (5, 13, 64, 16, 4), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 64, 16, 4),"ori_format": "ND"},
                    (0.0, 0.0, 0.0, 0.0),(1.0, 1.0, 1.0, 1.0)],
         "case_name": "BoundingBoxEncode_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (65535, 4), "dtype": "float16", "format": "ND", "ori_shape": (65535, 4),"ori_format": "ND"}, #x
                    {"shape": (65535, 4), "dtype": "float16", "format": "ND", "ori_shape": (65535, 4),"ori_format": "ND"}, #h
                    {"shape": (65535, 4), "dtype": "float16", "format": "ND", "ori_shape": (65535, 4),"ori_format": "ND"},
                    (1.0, 2.1, 0.0, 1.8),(2.1, 1.0, 3.6, 1.9)],
         "case_name": "BoundingBoxEncode_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (15, 32, 1, 1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32, 1, 1),"ori_format": "ND"}, #x
                    {"shape": (15, 32, 1, 1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32, 1, 1),"ori_format": "ND"}, #h
                    {"shape": (15, 32, 1, 1), "dtype": "float16", "format": "ND", "ori_shape": (15, 32, 1, 1),"ori_format": "ND"},
                    (0.0, 0.0, 0.0, 0.0),(1.0, 1.0, 1.0, 1.0)],
         "case_name": "BoundingBoxEncode_4",
         "expect": RuntimeError,
         "support_expect": True}
case5 = {"params": [{"shape": (3000000, 4), "dtype": "float16", "format": "ND", "ori_shape": (3000000, 4),"ori_format": "ND"}, #x
                    {"shape": (3000000, 4), "dtype": "float16", "format": "ND", "ori_shape": (3000000, 4),"ori_format": "ND"}, #h
                    {"shape": (3000000, 4), "dtype": "float16", "format": "ND", "ori_shape": (3000000, 4),"ori_format": "ND"},
                    (1.0, 2.1, 0.0, 1.8),(2.1, 1.0, 3.6, 1.9)],
         "case_name": "BoundingBoxEncode_5",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

# MOCK TEST
vals = {("tik.vgatherb", ): True}

def side_effects(*args):
    return vals[args]

with patch("te.platform.api_check_support", MagicMock(side_effect=side_effects)):
    ut_case.run("Ascend910",'BoundingBoxEncode_pre_static_BoundingBoxEncode_1')

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
