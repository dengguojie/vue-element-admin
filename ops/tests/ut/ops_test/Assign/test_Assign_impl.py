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

Assign ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("Assign", None, None)

case1 = {"params": [{"shape": (15, 32), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND"}, #x
                    {"shape": (15, 32), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND"}, #h
                    {"shape": (15, 32), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND"}, 
                    ],
         "case_name": "Assign_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (5, 13, 64), "dtype": "int16", "format": "ND", "ori_shape": (5, 13, 64),"ori_format": "ND"}, #x
                    {"shape": (5, 13, 64), "dtype": "int16", "format": "ND", "ori_shape": (5, 13, 64),"ori_format": "ND"}, #h
                    {"shape": (5, 13, 64), "dtype": "int16", "format": "ND", "ori_shape": (5, 13, 64),"ori_format": "ND"}, 
                    ],
         "case_name": "Assign_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (65535, 53,), "dtype": "int8", "format": "ND", "ori_shape": (65535, 53,),"ori_format": "ND"}, #x
                    {"shape": (65535, 53,), "dtype": "int8", "format": "ND", "ori_shape": (65535, 53,),"ori_format": "ND"}, #h
                    {"shape": (65535, 53,), "dtype": "int8", "format": "ND", "ori_shape": (65535, 53,),"ori_format": "ND"}, 
                    ],
         "case_name": "Assign_3",
         "expect": "success",
         "support_expect": True}
         
case4 = {"params": [{"shape": (128981,), "dtype": "float32", "format": "ND", "ori_shape": (128981,),"ori_format": "ND"}, #x
                    {"shape": (128981,), "dtype": "float32", "format": "ND", "ori_shape": (128981,),"ori_format": "ND"}, #h
                    {"shape": (128981,), "dtype": "float32", "format": "ND", "ori_shape": (128981,),"ori_format": "ND"}, 
                    ],
         "case_name": "Assign_4",
         "expect": "success",
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
 

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
