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

ArgMaxGradD ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("ArgMaxGradD", None, None)

# test data=4D dim=0~3  expect:success
cs4_0 = {"params": [{"shape": (6, 5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (5,16,16),"ori_format": "ND"}, # indices
                    {"shape": (5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5,16,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    0,
                    ],
         "case_name": "ArgMaxGradD_cs4_0",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

cs4_1 = {"params": [{"shape": (6, 5,16,16), "dtype": "float32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (6,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6,16,16),"ori_format": "ND"}, # indices
                    {"shape": (6,16,16), "dtype": "float32", "format": "ND", "ori_shape": (6,16,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "float32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    1,
                    ],
         "case_name": "ArgMaxGradD_cs4_1",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

cs4_2 = {"params": [{"shape": (6, 5,16,16), "dtype": "int8", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (6, 5,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # indices
                    {"shape": (6, 5,16), "dtype": "int8", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "int8", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    2,
                    ],
         "case_name": "ArgMaxGradD_cs4_2",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

cs4_3 = {"params": [{"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (6, 5,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # indices
                    {"shape": (6, 5,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    3,
                    ],
         "case_name": "ArgMaxGradD_cs4_3",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

# test data=4D dim=0~3  expect:RuntimeError
cs4_0F = {"params": [{"shape": (6, 5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5,16,16),"ori_format": "ND"}, # indices
                    {"shape": (5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5,16,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "float16", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    0,
                    ],
         "case_name": "ArgMaxGradD_cs4_0F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

cs4_1F = {"params": [{"shape": (6, 5,16,16), "dtype": "float32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (6,16,15), "dtype": "int32", "format": "ND", "ori_shape": (6,16,15),"ori_format": "ND"}, # indices
                    {"shape": (6,16,15), "dtype": "float32", "format": "ND", "ori_shape": (6,16,15),"ori_format": "ND"}, # updates
                    {"shape": (6, 4,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "float32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    1,
                    ],
         "case_name": "ArgMaxGradD_cs4_1F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

cs4_2F = {"params": [{"shape": (6, 5,16,16), "dtype": "uint8", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # indices
                    {"shape": (6, 5,16,16), "dtype": "uint8", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "uint8", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    2,
                    ],
         "case_name": "ArgMaxGradD_cs4_2F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

cs4_3F = {"params": [{"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (6, 5,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # indices
                    {"shape": (6, 5,2), "dtype": "float32", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    3,
                    ],
         "case_name": "ArgMaxGradD_cs4_3F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

cs4_4F = {"params": [{"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # var 
                    {"shape": (6, 5,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # indices
                    {"shape": (6, 5,16,4), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16),"ori_format": "ND"}, # updates
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # assist
                    {"shape": (6, 5,16,16), "dtype": "int32", "format": "ND", "ori_shape": (6, 5,16,16),"ori_format": "ND"}, # y
                    4,
                    ],
         "case_name": "ArgMaxGradD_cs4_3F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

# test data=2D dim=0~1  expect:success
cs2_0 = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # var 
                    {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # indices
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # updates
                    {"shape": (16,16), "dtype": "int32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # assist
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # y
                    0,
                    ],
         "case_name": "ArgMaxGradD_cs2_0",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

cs2_1 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # var 
                    {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # indices
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # updates
                    {"shape": (16,16), "dtype": "int32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # assist
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # y
                    1,
                    ],
         "case_name": "ArgMaxGradD_cs2_1",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

# test data=2D dim=0~1  expect:RuntimeError
cs2_0F = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # var 
                    {"shape": (15,), "dtype": "int32", "format": "ND", "ori_shape": (15,),"ori_format": "ND"}, # indices
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # updates
                    {"shape": (16,16), "dtype": "int32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # assist
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # y
                    0,
                    ],
         "case_name": "ArgMaxGradD_cs2_0F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

cs2_1F = {"params": [{"shape": (16,16), "dtype": "uint8", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # var 
                    {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # indices
                    {"shape": (16,), "dtype": "uint8", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # updates
                    {"shape": (16,16), "dtype": "int32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # assist
                    {"shape": (16,16), "dtype": "uint8", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}, # y
                    0,
                    ],
         "case_name": "ArgMaxGradD_cs2_1F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

# test data=1D dim=0  expect:success
cs1_0 = {"params": [{"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # var 
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, # indices
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, # updates
                    {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # assist
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # y
                    0,
                    ],
         "case_name": "ArgMaxGradD_cs2_0",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

# test data=1D dim=0  expect:RuntimeError
cs1_0F = {"params": [{"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # var 
                    {"shape": (2,16), "dtype": "int32", "format": "ND", "ori_shape": (2,16),"ori_format": "ND"}, # indices
                    {"shape": (2,16), "dtype": "int32", "format": "ND", "ori_shape": (2,16),"ori_format": "ND"}, # updates
                    {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # assist
                    {"shape": (16,), "dtype": "int32", "format": "ND", "ori_shape": (16,),"ori_format": "ND"}, # y
                    0,
                    ],
         "case_name": "ArgMaxGradD_cs2_0F",
         "expect": RuntimeError,
         "format_expect": ["ND"],
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310"], cs4_0)
ut_case.add_case(["Ascend910","Ascend310"], cs4_1)
ut_case.add_case(["Ascend910","Ascend310"], cs4_2)
ut_case.add_case(["Ascend910"], cs4_3)
ut_case.add_case(["Ascend910","Ascend310"], cs4_0F)
ut_case.add_case(["Ascend910","Ascend310"], cs4_1F)
ut_case.add_case(["Ascend910","Ascend310"], cs4_2F)
ut_case.add_case(["Ascend910","Ascend310"], cs4_3F)
ut_case.add_case(["Ascend910","Ascend310"], cs4_4F)
ut_case.add_case(["Ascend910","Ascend310"], cs2_0)
ut_case.add_case(["Ascend910","Ascend310"], cs2_1)
ut_case.add_case(["Ascend910","Ascend310"], cs2_0F)
ut_case.add_case(["Ascend910","Ascend310"], cs2_1F)
ut_case.add_case(["Ascend910","Ascend310"], cs1_0)
ut_case.add_case(["Ascend910","Ascend310"], cs1_0F)
