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

Mul ut case
"""
# pylint: disable=unused-import, pointless-string-statement
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
from te import tvm
ut_case = OpUT("Mul", None, None)

case1 = {"params": [{"shape": (8192, 1), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (8192, 1), "ori_format": "NHWC"},
                    {"shape": (8192, 100), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (8192, 100), "ori_format": "NHWC"},
                    {"shape": (8192, 1), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (8192, 1), "ori_format": "NHWC"}],
         "case_name": "mul_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (10241,), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (10241,), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (10241,), "dtype": "int8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "int8", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "int8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (10241,), "dtype": "uint8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "uint8", "format": "NHWC",
                     "ori_shape": (10, 10241), "ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "uint8", "format": "NHWC",
                     "ori_shape": (10241,), "ori_format": "NHWC"}
                    ],
         "case_name": "mul_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (3, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"},
                    {"shape": (3, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"}],
         "case_name": "mul_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"},
                    {"shape": (1,), "dtype": "float16", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 16, 16), "ori_format": "FRACTAL_NZ"}],
         "case_name": "mul_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

def test_mul_compute_nz_nd_ubfusion_1(test_arg):
    from impl.mul import mul_compute
    x = tvm.placeholder((15, 512, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (8192, 240)})
    y = tvm.placeholder((1, 1, 1, 240), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (240,)})
    output = {"shape": (15, 512, 16, 16), "dtype": "float16", "ori_shape": (8192, 240), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False)  

def test_mul_compute_nz_nd_ubfusion_2(test_arg):
    from impl.mul import mul_compute
    x = tvm.placeholder((2, 16, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (256, 32)})
    y = tvm.placeholder((256, 1), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (256, 1)})
    output = {"shape": (2, 16, 16, 16), "dtype": "float16", "ori_shape": (256, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False)

def test_mul_compute_nz_nd_ubfusion_3(test_arg):
    from impl.mul import mul_compute
    try:
        x = tvm.placeholder((4, 2, 4, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (4, 64, 32)})
        y = tvm.placeholder((4, 1, 1), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (4, 1, 1)})
        output = {"shape": (4, 2, 4, 16, 16), "dtype": "float16", "ori_shape": (4, 64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
        mul_compute(x, y, output, False)
    except RuntimeError as e:
        print("test_mul_compute_nz_nd_ubfusion_3_ivalid success")

def test_mul_compute_nd_nz_ubfusion_1(test_arg):
    from impl.mul import mul_compute
    x = tvm.placeholder((1, 1, 1, 240), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (240,)})
    y = tvm.placeholder((15, 512, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (8192, 240)})
    output = {"shape": (15, 512, 16, 16), "dtype": "float16", "ori_shape": (8192, 240), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False) 

def test_mul_compute_nd_nz_ubfusion_2(test_arg):
    from impl.mul import mul_compute
    x = tvm.placeholder((256, 1), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (256, 1)})
    y = tvm.placeholder((2, 16, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (256, 32)})
    output = {"shape": (2, 16, 16, 16), "dtype": "float16", "ori_shape": (256, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False)

def test_mul_compute_nd_nz_ubfusion_3(test_arg):
    from impl.mul import mul_compute
    try:
        x = tvm.placeholder((4, 1, 1), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (4, 1, 1)})
        y = tvm.placeholder((4, 2, 4, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (4, 64, 32)})
        output = {"shape": (4, 2, 4, 16, 16), "dtype": "float16", "ori_shape": (4, 64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
        mul_compute(x, y, output, False)
    except RuntimeError as e:
        print("test_mul_compute_nd_nz_ubfusion_3_ivalid success")

# pylint: disable=unused-argument
def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.mul import op_select_format
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (20, 28, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (20, 28, 16), "ori_format": "ND"})
    op_select_format({"shape": (1, 1, 1), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (1, 1, 1), "ori_format": "NHWC"},
                     {"shape": (96, 1, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (96, 56, 56, 8), "ori_format": "NHWC"},
                     {"shape": (96, 1, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (96, 56, 56, 8), "ori_format": "NHWC"})
    op_select_format({"shape": (), "dtype": "float32", "format": "NHWC",
                      "ori_shape": (), "ori_format": "NHWC"},
                     {"shape": (96, 256), "dtype": "float32", "format": "FRACTAL_NZ",
                      "ori_shape": (96, 256), "ori_format": "NHWC"},
                     {"shape": (96, 256), "dtype": "float32", "format": "FRACTAL_NZ",
                      "ori_shape": (96, 256), "ori_format": "NHWC"})
    op_select_format({"shape": (25, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z",
                      "ori_shape": (6, 1, 5, 5), "ori_format": "NCHW"},
                     {"shape": (), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (), "ori_format": "NCHW"},
                     {"shape": (25, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z",
                      "ori_shape": (6, 1, 5, 5), "ori_format": "NCHW"})
    op_select_format({"shape": (512,), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (512,), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"})
    op_select_format({"shape": (33, 17, 3, 5, 3), "dtype": "float16", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float16", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "int32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "int32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "int32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "uint8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "uint8", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "uint8", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (3, 32, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (3, 32, 32), "ori_format": "ND"},
                     {"shape": (3, 32, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (3, 32, 32), "ori_format": "ND"})
    op_select_format({"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (1,), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"})
    op_select_format({"shape": (-1, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (-1, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (-1, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (-1, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1})
    op_select_format({"shape": (32, 2, 2, 16, 16), "dtype": "float16", "format": "NDHWC", "ori_shape": (32, 2, 2, 16, 16),
                      "ori_format": "NDHWC", "sub_format" : 0},
                     {"shape": (16,), "dtype": "float16", "format": "NDHWC", "ori_shape": (16,),
                      "ori_format": "NDHWC", "sub_format" : 0},
                     {"shape": (32, 2, 2, 16, 16), "dtype": "float16", "format": "NDHWC", "ori_shape": (32, 2, 2, 16, 16),
                      "ori_format": "NDHWC", "sub_format" : 0})
    op_select_format({"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (240,), "dtype": "float16", "format": "ND", "ori_shape": (240,),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240),
                      "ori_format": "ND", "sub_format" : 0})
    op_select_format({"shape": (240,), "dtype": "float16", "format": "ND", "ori_shape": (240,),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240),
                      "ori_format": "ND", "sub_format" : 0})
    op_select_format({"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (256, 1), "dtype": "float16", "format": "ND", "ori_shape": (256, 1),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32),
                      "ori_format": "ND", "sub_format" : 0})
    op_select_format({"shape": (256, 1), "dtype": "float16", "format": "ND", "ori_shape": (256, 1),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32),
                      "ori_format": "ND", "sub_format" : 0},
                     {"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32),
                      "ori_format": "ND", "sub_format" : 0})

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_mul_compute_nz_nd_ubfusion_1)
ut_case.add_cust_test_func(test_func=test_mul_compute_nz_nd_ubfusion_2)
ut_case.add_cust_test_func(test_func=test_mul_compute_nz_nd_ubfusion_3)
ut_case.add_cust_test_func(test_func=test_mul_compute_nd_nz_ubfusion_1)
ut_case.add_cust_test_func(test_func=test_mul_compute_nd_nz_ubfusion_2)
ut_case.add_cust_test_func(test_func=test_mul_compute_nd_nz_ubfusion_3)

"""
The ca_model of CI is faulty.Related cases are commented out temporaily.
def calc_expect_func(input_a, input_b, output_y):
    return np.multiply(input_a["value"], input_b["value"]).astype(input_a["dtype"])

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                     "ori_shape": (92, 1), "shape": (92, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                     "ori_shape": (92, 100), "shape": (92, 100), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                     "ori_shape": (92, 100), "shape": (92, 100), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1024, 3), "shape": (1024, 3), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (10, 11, 1), "shape": (10, 11, 1), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (3, 3, 144, 1), "shape": (3, 3, 144, 1), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
"""


if __name__ == '__main__':
    ut_case.run("Ascend910A")
