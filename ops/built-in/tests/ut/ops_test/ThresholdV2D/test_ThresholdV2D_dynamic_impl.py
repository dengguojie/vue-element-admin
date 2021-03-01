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

TileWithAxis ut case
"""
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("ThresholdV2D", "impl.dynamic.threshold_v2_d", "threshold_v2_d")

case1 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "ND", 
                     "ori_shape": (5, 13, 4), "ori_format": "ND", "range":[(1, None), (1, None), (1, None)]}, #x
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "ND", 
                     "ori_shape": (5, 13, 4),"ori_format": "ND", "range":[(1, None), (1, None), (1, None)]},
                    4.5, 3.2
                    ],
         "case_name": "ThresholdV2D_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1), "dtype": "float32", "format": "ND", 
                     "ori_shape": (65, 75),"ori_format": "ND", "range":[(1, None), (1, None)]}, #x
                    {"shape": (-1, -1), "dtype": "float32", "format": "ND", 
                     "ori_shape": (65, 75),"ori_format": "ND", "range":[(1, None), (1, None)]},
                    5.6, 8.6
                    ],
         "case_name": "ThresholdV2D_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (13, 7, 5, 3), 
                     "ori_format": "ND", "range":[(1, None), (1, None), (1, None), (1,None)]}, #x
                    {"shape": (-1, -1, -1, -1), "dtype": "int8", "format": "ND", "ori_shape": (13, 7, 5, 3), 
                     "ori_format": "ND", "range":[(1, None), (1, None), (1, None), (1,None)]},
                    10.5, 8.6
                    ],
         "case_name": "ThresholdV2D_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND", 
                     "ori_shape": (160000, 16),"ori_format": "ND", "range":[(1, None), (1, None)]}, #x
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", 
                     "ori_shape": (160000, 16),"ori_format": "ND", "range":[(1, None), (1, None)]},
                    6.6, 7.4
                    ],
         "case_name": "ThresholdV2D_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (-1,), "dtype": "float32", "format": "ND", 
                     "ori_shape": (459999,),"ori_format": "ND", "range":[(1, None)]}, #x
                    {"shape": (-1,), "dtype": "float32", "format": "ND", 
                     "ori_shape": (459999,),"ori_format": "ND", "range":[(1, None)]},
                    5.5, 3.1
                    ],
         "case_name": "ThresholdV2D_5",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
