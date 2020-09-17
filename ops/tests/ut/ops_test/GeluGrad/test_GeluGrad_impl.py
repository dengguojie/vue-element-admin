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

GeluGrad ut case
"""
from op_test_frame.ut import OpUT
import sys
import sys
import time
import unittest
ut_case = OpUT("GeluGrad", None, None)

case1 = {"params": [{"shape": (5, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"}, #x
                    {"shape": (5, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    {"shape": (5, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    {"shape": (5, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"}, #x
                    {"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"}, #h
                    {"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    {"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"}, #x
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"}, #h
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"}, #x
                    {"shape": (3, 128, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 128, 32),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"}, #x
                    {"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"}, #h
                    {"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"},
                    {"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"},
                    ],
         "case_name": "GeluGrad_5",
         "expect": RuntimeError,
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
