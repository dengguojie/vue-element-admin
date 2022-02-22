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

Shrink ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("Shrink", "impl.dynamic.shrink", "shrink")
# dtype=float16
case1 = {"params": [
                {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),
                "ori_format": "ND","range":[(1,None),(1,None)]}, # input_x
                {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),
                "ori_format": "ND","range":[(1,None),(1,None)]}, # output_x            
            ],
            "case_name": "Shrink_float16_dim2_1",
            "expect": "success",
            "support_expect": True
        }

case2 = {"params": [
                {"shape": (-1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1),
                "ori_format": "ND","range":[(1,None),(1,None),(1,None)]}, # input_x
                {"shape": (-1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1),
                "ori_format": "ND","range":[(1,None),(1,None),(1,None)]}, # output_x            
            ],
            "case_name": "Shrink_float16_dim3_2",
            "expect": "success",
            "support_expect": True
        }

case3 = {"params": [
                {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", 
                "ori_shape": (-1, -1, -1, -1),"ori_format": "ND","range":[(1,None),(1,None),(1,None),(1,None)]},
                {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", 
                "ori_shape": (-1, -1, -1, -1),"ori_format": "ND","range":[(1,None),(1,None),(1,None),(1,None)]},          
            ],
            "case_name": "Shrink_float16_dim4_3",
            "expect": "success",
            "support_expect": True
        }
# dtype=float32
case4 = {"params": [
                {"shape": (-1, -1), "dtype": "float32", "format": "ND", 
                "ori_shape": (-1, -1),"ori_format": "ND","range":[(1,None),(1,None)]}, # input_x
                {"shape": (-1, -1), "dtype": "float32", "format": "ND", 
                "ori_shape": (-1, -1),"ori_format": "ND","range":[(1,None),(1,None)]}, # output_x            
            ],
            "case_name": "Shrink_float32_dim2_4",
            "expect": "success",
            "support_expect": True
        }

case5 = {"params": [
                {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1),
                "ori_format": "ND","range":[(1,None),(1,None),(1,None)]}, # input_x
                {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1),
                "ori_format": "ND","range":[(1,None),(1,None),(1,None)]}, # output_x            
            ],
            "case_name": "Shrink_float32_dim3_5",
            "expect": "success",
            "support_expect": True
        }

case6 = {"params": [
                {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", 
                "ori_shape": (-1, -1, -1, -1),"ori_format": "ND","range":[(1,None),(1,None),(1,None),(1,None)]}, 
                {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", 
                "ori_shape": (-1, -1, -1, -1),"ori_format": "ND","range":[(1,None),(1,None),(1,None),(1,None)]},         
            ],
            "case_name": "Shrink_float32_dim4_6",
            "expect": "success",
            "support_expect": True
        }

# static case
case7 = {"params": [
                {"shape": (7, 8), "dtype": "float16", "format": "ND", 
                "ori_shape": (7, 8),"ori_format": "ND"}, # input_x
                {"shape": (7, 8), "dtype": "float16", "format": "ND", 
                "ori_shape": (7, 8),"ori_format": "ND"}, # output_x            
            ],
            "case_name": "Shrink_float16_dim2_static_1",
            "expect": "success",
            "support_expect": True
        }
# add case
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)



if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910"])
    exit(0)
