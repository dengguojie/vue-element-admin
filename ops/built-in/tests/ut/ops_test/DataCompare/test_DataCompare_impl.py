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

DataCompare ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("DataCompare", None, None)

case1 = {"params": [{"shape": (5, 8,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5, 8,16,16),"ori_format": "ND"},
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5, 8,16,16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001,0.0001,
                    ],
         "case_name": "DataCompare_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND", "ori_shape": (30000, 1),"ori_format": "ND"}, #x
                    {"shape": (30000, 1), "dtype": "float32", "format": "ND", "ori_shape": (30000, 1),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001,0.0001,
                    ],
         "case_name": "DataCompare_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2,16,32), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #x
                    {"shape": (2,16,32), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001,0.0001,
                    ],
         "case_name": "DataCompare_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2,10,1028,1,16), "dtype": "float16", "format": "ND", "ori_shape": (2,10,1028,1,16),"ori_format": "ND"}, #x
                    {"shape": (2,10,1028,1,16), "dtype": "float16", "format": "ND", "ori_shape": (2,10,1028,1,16),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.001,0.001,
                    ],
         "case_name": "DataCompare_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case1)
ut_case.add_case(["Ascend910","Ascend310P3"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case4)
