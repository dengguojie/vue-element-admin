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

BoundingBoxDecode ut case
"""
import tbe
from unittest.mock import patch
from unittest.mock import MagicMock
from op_test_frame.ut import OpUT
ut_case = OpUT("BoundingBoxDecode", "impl.dynamic.bounding_box_decode", "bounding_box_decode")

case1 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND",
                     "range":[(1,None),(1,None),(1,None),(4,4)]}, #rois
                    {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND",
                     "range":[(1,None),(1,None),(1,None),(4,4)]}, #deltas
                    {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND",
                     "range":[(1,None),(1,None),(1,None),(4,4)]}, 
                    (1.0, 2.1, 0.0, 1.8),(2.1, 1.0, 3.6, 1.9),None,0.016,
                    ],
         "case_name": "BoundingBoxDecode_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),"ori_format": "ND",
                     "range":[(1,None),(1,None),(1,None),(1,None),(4,4)]}, #x
                    {"shape": (-1, -1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),"ori_format": "ND",
                     "range":[(1,None),(1,None),(1,None),(1,None),(4,4)]},
                    {"shape": (-1, -1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),"ori_format": "ND",
                     "range":[(1,None),(1,None),(1,None),(1,None),(4,4)]},
                    (0.0, 0.0, 0.0, 0.0),(1.0, 1.0, 1.0, 1.0),None,0.016,
                    ],
         "case_name": "BoundingBoxDecode_dynamic_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND", 
                    "range":[(1,None),(4,4)]}, #x
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND", 
                    "range":[(1,None),(4,4)]}, #h
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND", 
                    "range":[(1,None),(4,4)]},
                    (1.0, 2.1, 0.0, 1.8),(2.1, 1.0, 3.6, 1.9),None,0.016,
                    ],
         "case_name": "BoundingBoxDecode_dynamic_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND", 
                     "range":[(1,None),(4,4)]}, #x
                    {"shape": (-1, 4), "dtype": "float32", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND", 
                     "range":[(1,None),(4,4)]}, #h
                    {"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND", 
                     "range":[(1,None),(4,4)]},
                    (1.0, 2.1, 0.0, 1.8),(2.1, 1.0, 3.6, 1.9),None,0.016,
                    ],
         "case_name": "BoundingBoxDecode_dynamic_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND", 
                     "range":[(1,None),(4,4)]}, #x
                    {"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND", 
                     "range":[(1,None),(4,4)]}, #h
                    {"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND", 
                     "range":[(1,None),(4,4)]},
                    (1.0, 2.1, 0.0, 1.8),(2.1, 1.0, 3.6, 1.9),None,0.016,
                    ],
         "case_name": "BoundingBoxDecode_dynamic_5",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

# MOCK TEST
vals = {("tik.vgatherb", ): True}

def side_effects(*args):
    return vals[args]

with patch("impl.util.platform_adapter.tbe_platform.api_check_support", MagicMock(side_effect=side_effects)):
    ut_case.run("Ascend910",'BoundingBoxDecode_dynamic_BoundingBoxDecode_dynamic_1')

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
