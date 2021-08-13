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

if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend710"])
    exit(0)
