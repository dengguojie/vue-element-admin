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

BatchMatmul ut case
"""
from op_test_frame.ut import OpUT
import sys
import time
import unittest
ut_case = OpUT("BatchMatmulV2", None, None)

from tbe import tvm
from impl.batch_matmul_v2 import check_supported
from impl.batch_matmul_v2 import get_op_support_info

case1 = {"params": [{"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmul_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 32, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 32, 96),"ori_format": "NHWC"}, #x
                    {"shape": (3, 96, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 64),"ori_format": "NHWC"}, #h
                    {"shape": (96,), "dtype": "float32", "format": "NHWC", "ori_shape": (96,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 96),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"}, #x
                    {"shape": (3, 128, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 128, 32),"ori_format": "NHWC"},
                    {"shape": (128, ), "dtype": "float16", "format": "NHWC", "ori_shape": (128,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 112, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 112, 64),"ori_format": "ND"}, #x
                    {"shape": (3, 64, 112), "dtype": "float32", "format": "ND", "ori_shape": (3, 64, 112),"ori_format": "ND"}, #h
                    {"shape": (64,), "dtype": "float32", "format": "ND", "ori_shape": (64,),"ori_format": "ND"},
                     None,
                    {"shape": (3, 112, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 112, 64),"ori_format": "ND"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (112, 64), "dtype": "float32", "format": "ND", "ori_shape": (112, 64),"ori_format": "ND"},
                    {"shape": (64, 112), "dtype": "float32", "format": "ND", "ori_shape": (64, 112),"ori_format": "ND"},
                    {"shape": (64,), "dtype": "float32", "format": "ND", "ori_shape": (64,),"ori_format": "ND"},
                     None,
                    {"shape": (64, 64), "dtype": "float32", "format": "ND", "ori_shape": (64, 64),"ori_format": "ND"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (4, 1, 13, 11, 1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (4, 1, 13, 11, 1, 2),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (2, 1),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                     None,
                    {"shape": (4, 1, 13, 11, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (4, 1, 13, 11, 1, 1),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_5",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (4, 3, 96, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (4, 3,96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (3, 64, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float32", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmul_support_check1"}

case8 = {"params": [{"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (1, 1, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float32", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    False,True
                    ],
         "case_name": "BatchMatmul_support_check2"}

case9 = {"params": [{"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (1, 32, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 32, 1),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float32", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True, False
                    ],
         "case_name": "BatchMatmul_support_check3"}

case10 = {"params": [{"shape": (3, 32, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 64),"ori_format": "NHWC"}, #x
                     {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 64, 96),"ori_format": "NHWC"},
                     {"shape": (64,), "dtype": "float16", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                      None,
                     {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                     True,False
                     ],
          "case_name": "BatchMatmul_support_check4"}

case11 = {"params": [{"shape": (8, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (8, 1, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 1, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (8, 4, 1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 4, 32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_11",
         "expect": "success",
         "support_expect": True}

case12 = {"params": [{"shape": (2, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (2, 4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_12",
         "expect": "success",
         "support_expect": True}

case13 = {"params": [{"shape": (2, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_13",
         "expect": "success",
         "support_expect": True}

case14 = {"params": [{"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 64),"ori_format": "ND"},
                     {"shape": (4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (4, 1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_14",
         "expect": "success",
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend920A"], case14)

def test_split_batch_matmul_v2(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    get_op_support_info(x1, x2, trans_a=True)

def test_split_batch_matmul_v2_1(test_arg):
    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16, 32), "ori_shape": (16, 16, 32)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32), "ori_shape": (16, 32)}
    get_op_support_info(x1, x2, trans_b=True)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2_1)


def test_op_check_supported(test_arg):
    def _test_supported(case):
        input_x, input_y, bias, offset_w, output_z, trans_a, trans_b = case["params"]
        try:
            print(check_supported(input_x, input_y, bias, offset_w, output_z, trans_a, trans_b, kernel_name="batch_matmul"))
        except RuntimeError:
            print("The case is not supported!")
            pass

    _test_supported(case7)
    _test_supported(case8)
    _test_supported(case9)
    _test_supported(case10)

ut_case.add_cust_test_func(test_func=test_op_check_supported)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)
