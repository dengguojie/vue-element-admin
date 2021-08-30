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

TransposeD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("TransposeD", None, None)

case1 = {"params": [{"shape": (2, 3, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 16),"ori_format": "ND"}, #x
                    {"shape": (2, 3, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 16),"ori_format": "ND"},
                    (0, 2, 1, 3),
                    ],
         "case_name": "TransposeD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 4, 5, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 5, 32),"ori_format": "ND"}, #x
                    {"shape": (3, 4, 5, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 5, 32),"ori_format": "ND"},
                    (0, 2, 1, 3),
                    ],
         "case_name": "TransposeD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (4032, 64), "dtype": "int32", "format": "ND", "ori_shape": (4032, 64),"ori_format": "ND"}, #x
                    {"shape": (4032, 64), "dtype": "int32", "format": "ND", "ori_shape": (4032, 64),"ori_format": "ND"},
                    (1, 0),
                    ],
         "case_name": "TransposeD_3",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (1,32,2,2,1920,960), "dtype": "float16", "format": "ND", "ori_shape": (1,32,2,2,1920,960),"ori_format": "ND"}, #x
                    {"shape": (1,32,1920,2,960,2), "dtype": "int32", "format": "ND", "ori_shape":(1,32,1920,2,960,2),"ori_format": "ND"},
                    (0,1,4,2,5,3),
                    ],
         "case_name": "TransposeD_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (1,32,2,2,960,960), "dtype": "float16", "format": "ND", "ori_shape": (1,32,2,2,960,960),"ori_format": "ND"}, #x
                    {"shape": (1,32,960,2,960,2), "dtype": "int32", "format": "ND", "ori_shape":(1,32,960,2,960,2),"ori_format": "ND"},
                    (0,1,4,2,5,3),
                    ],
         "case_name": "TransposeD_10",
         "expect": "success",
         "support_expect": True}


case11 = {"params": [{"shape": (1,3,80,80,85), "dtype": "float32", "format": "ND", "ori_shape": (1,3,80,80,85),"ori_format": "ND"}, #x
                    {"shape": (1,3,80,80,85), "dtype": "float32", "format": "ND", "ori_shape":(1,3,85,80,80),"ori_format": "ND"},
                    (0,1,3,4,2),
                    ],
         "case_name": "TransposeD_11",
         "expect": "success",
         "support_expect": True}


case12 = {"params": [{"shape": (1,32,2,2,1024,768), "dtype": "float16", "format": "ND", "ori_shape": (1,32,2,2,1024,768),"ori_format": "ND"}, #x
                    {"shape": (1,32,1024,2,768,2), "dtype": "float16", "format": "ND", "ori_shape":(1,32,2,2,1024,768),"ori_format": "ND"},
                    (0,1,4,2,5,3),
                    ],
         "case_name": "TransposeD_12",
         "expect": "success",
         "support_expect": True}

case13 = {"params": [{"shape": (0,3,4,1,2), "dtype": "float16", "format": "ND", "ori_shape":(0,3,4,1,2),"ori_format": "ND"}, #x
                    {"shape": (0,3,4,1,2), "dtype": "float16", "format": "ND", "ori_shape":(0,3,4,1,2),"ori_format": "ND"},
                    (0,3,4,1,2),
                    ],
         "case_name": "TransposeD_13",
         "expect": RuntimeError,
         "support_expect": True}

case4 = {"params": [{"shape": (1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"},
                    (1, 0, 2),
                    ],
         "case_name": "TransposeD_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"},
                    (1, 0, 5, 3),
                    ],
         "case_name": "TransposeD_5",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (1, 216, 216, 64), "dtype": "float32", "format": "ND", "ori_shape": (1, 216, 216, 64),"ori_format": "ND"}, #x
                    {"shape": (1, 64, 216, 216), "dtype": "float32", "format": "ND", "ori_shape": (1, 64, 216, 216),"ori_format": "ND"},
                    (0, 3, 1, 2),
                    ],
         "case_name": "TransposeD_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (1, 432, 432, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 432, 432, 2),"ori_format": "ND"}, #x
                    {"shape": (1, 432, 2, 432), "dtype": "float32", "format": "ND", "ori_shape": (1, 432, 2, 432),"ori_format": "ND"},
                    (0, 1, 3, 2),
                    ],
         "case_name": "TransposeD_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (1, 256, 108, 108), "dtype": "float32", "format": "ND", "ori_shape": (1, 256, 108, 108),"ori_format": "ND"}, #x
                    {"shape": (1, 108, 108, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 108, 108, 256),"ori_format": "ND"},
                    (0, 2, 3, 1),
                    ],
         "case_name": "TransposeD_8",
         "expect": RuntimeError,
         "support_expect": True}

#TODO  customize test function 
def test_op_get_op_support_info(test_arg): 
    from impl.transpose_d import get_op_support_info
    get_op_support_info({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 3, 4, 1, 2],
                        "get_op_support_info_case1")     
    
    get_op_support_info({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 2, 3, 1],
                        "get_op_support_info_case2")   
    
    get_op_support_info({"shape": (16, 16, 16, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [3, 2, 0, 1],
                        "get_op_support_info_case3")
    
    get_op_support_info({"shape": (1, 16, 1, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (1, 16, 1, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [3, 2, 0, 1],
                        "get_op_support_info_case4")   
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [3, 2, 0, 1],
                        "get_op_support_info_case5")   
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 3, 1, 2, 4],
                        "get_op_support_info_case6")    

    get_op_support_info({"shape": (16, 1, 16, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 1, 16, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [1, 2, 3, 0],
                        "get_op_support_info_case7") 

    get_op_support_info({"shape": (16, 1, 16, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 1, 16, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [3, 2, 1, 0],
                        "get_op_support_info_case8") 

    get_op_support_info({"shape": (1, 16, 1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (1, 16, 1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [3, 0, 2, 1],
                        "get_op_support_info_case9") 

    get_op_support_info({"shape": (1, 1, 1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (1, 1, 1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 2, 1],
                        "get_op_support_info_case10") 
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 2, 1],
                        "get_op_support_info_case11")   
    
    get_op_support_info({"shape": (1, 1, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (1, 1, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [1, 0, 2],
                        "get_op_support_info_case12")     
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [1, 0, 2],
                        "get_op_support_info_case13")     
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 1, 3, 2],
                        "get_op_support_info_case14")    
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 1, 2, 4, 3],
                        "get_op_support_info_case15")      
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [2, 0, 1],
                        "get_op_support_info_case16")  

    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 2, 3, 4, 1],
                        "get_op_support_info_case17")    
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [0, 4, 1, 2, 3],
                        "get_op_support_info_case18")      
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        [8, 8, 8],
                        "get_op_support_info_case19")    
    
    get_op_support_info({"shape": (16, 16, 16, 2), "dtype": "float16", "format": "NCDHW", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16, 16, 2), "dtype": "float16", "format": "NCDHW", "ori_shape": (1, 1), "ori_format": "ND"},
                        [8, 8, 8],
                        "get_op_support_info_case20")  

ut_case.add_cust_test_func(test_func=test_op_get_op_support_info)
# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case8)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case9)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case10)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case11)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case12)
ut_case.add_case(["Ascend910A","Ascend710"], case13)



if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend710"])
    exit(0)
