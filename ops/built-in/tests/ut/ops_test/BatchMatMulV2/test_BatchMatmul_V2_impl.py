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
import json
from unittest.mock import MagicMock
from unittest.mock import patch
from op_test_frame.ut import OpUT
ut_case = OpUT("BatchMatMulV2", "impl.batch_matmul_v2", "batch_matmul_v2")

from tbe import tvm
from impl.batch_matmul_v2 import get_op_support_info
from test_bmmv2_mock_case import *
from case_nd_in_nd_out_bmm import cases as nd_cases_bmm

vals = {("CORE_NUM", ): 48,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 524288,
        ("L0C_SIZE", ): 131072,
        ("Intrinsic_fix_pipe_l0c2out",): True,
        ("Intrinsic_fix_pipe_unit_list",): True,
        ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True,
        ("Intrinsic_vconv", "s322f32"): True
        }
def side_effects(*args):
    return vals[args]

case1 = {"params": [{"shape": (3, 96, 32), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmul_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 32, 96), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3, 32, 96),"ori_format": "NHWC"},
                    {"shape": (3, 96, 64), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3, 96, 64),"ori_format": "NHWC"},
                    {"shape": (96,), "dtype": "float32", "format": "NHWC", "ori_shape": (96,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 96), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3, 96, 96),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    {"shape": (3, 128, 32), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3, 128, 32),"ori_format": "NHWC"},
                    {"shape": (128, ), "dtype": "float16", "format": "NHWC", "ori_shape": (128,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    True,True,
                    ],
         "case_name": "BatchMatmul_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 112, 64), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 112, 64),"ori_format": "ND"},
                    {"shape": (3, 64, 112), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 64, 112),"ori_format": "ND"},
                    {"shape": (64,), "dtype": "float32", "format": "ND", "ori_shape": (64,),"ori_format": "ND"},
                     None,
                    {"shape": (3, 112, 64), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 112, 64),"ori_format": "ND"},
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

case6 = {"params": [{"shape": (4, 1, 13, 11, 1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
                     "ori_shape": (4, 1, 13, 11, 1, 2),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_Z",
                     "ori_shape": (2, 1),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                     None,
                    {"shape": (4, 1, 13, 11, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                     "ori_shape": (4, 1, 13, 11, 1, 1),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_5",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (4, 3, 96, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (4, 3,96, 32),"ori_format": "NHWC"},
                    {"shape": (3, 64, 96), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (64,),"ori_format": "NHWC"},
                    {"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmul_support_check1"}

case8 = {"params": [{"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3, 96, 32),"ori_format": "NHWC"},
                    {"shape": (1, 1, 96), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (1, 1, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float32", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                    {"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    False,True
                    ],
         "case_name": "BatchMatmul_support_check2"}

case9 = {"params": [{"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3, 96, 32),"ori_format": "NHWC"},
                    {"shape": (1, 32, 1), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (1, 32, 1),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float32", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                    {"shape": (3, 96, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True, False
                    ],
         "case_name": "BatchMatmul_support_check3"}

case10 = {"params": [{"shape": (3, 32, 64), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (3, 32, 64),"ori_format": "NHWC"},
                     {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (3, 64, 96),"ori_format": "NHWC"},
                     {"shape": (64,), "dtype": "float16", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                     {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                     True,False
                     ],
          "case_name": "BatchMatmul_support_check4"}

case15 = {"params": [{"shape": (0, 10), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (0, 10),"ori_format": "ND"},
                     {"shape": (10, 20), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (10, 20),"ori_format": "ND"},
                     None,
                     {"shape": (0, 20), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (0, 20),"ori_format": "ND"},
                     False,False,
                     ],
          "case_name": "BatchMatmul_support_check5"}

case11 = {"params": [{"shape": (8, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (8, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (8, 1, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (8, 1, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (8, 4, 1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (8, 4, 32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_11",
         "expect": "success",
         "support_expect": True}

case12 = {"params": [{"shape": (2, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (2, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (2, 4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (2, 4, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_12",
         "expect": "success",
         "support_expect": True}

case13 = {"params": [{"shape": (2, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (2, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (4, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_13",
         "expect": "success",
         "support_expect": True}

case14 = {"params": [{"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (4, 32, 64),"ori_format": "ND"},
                     {"shape": (4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (4, 64, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (4, 1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (4, 32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_14",
         "expect": "success",
         "support_expect": True}

case15 = {"params": [{"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (4, 32, 64),"ori_format": "ND"},
                     {"shape": (4, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (4, 64, 49),"ori_format": "ND"},
                     {"shape": (49,), "dtype": "float32", "format": "ND", "ori_shape": (49,),"ori_format": "ND"},
                    {"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (4, 32, 49),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_15",
         "expect": "success",
         "support_expect": True}

case16 = {"params": [{"shape": (96, 32), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (96, 32),"ori_format": "NHWC"},
                    {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmulv2_16",
         "expect": "success",
         "support_expect": True}

case17 = {"params": [{"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    {"shape": (64, 96), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmulv2_17",
         "expect": "success",
         "support_expect": True}

case18 = {"params": [{"shape": (128, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (16, 2048),"ori_format": "ND"},
                    {"shape": (36, 128, 16, 16), "dtype": "float16", "format": "FRACTAL_Z",
                     "ori_shape": (576, 2048),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (1, 36, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 36, 16, 16),"ori_format": "FRACTAL_NZ"},
                    False,True
                    ],
         "case_name": "BatchMatmulv2_18",
         "expect": "success",
         "support_expect": True}

case19 = {"params": [{"shape": (128, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (16, 2048),"ori_format": "ND"},
                    {"shape": (36, 128, 16, 16), "dtype": "float16", "format": "FRACTAL_ZN_RNN",
                     "ori_shape": (576, 2048),"ori_format": "ND"},
                    None,
                    {"shape": (1, 36, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 36, 16, 16),"ori_format": "FRACTAL_NZ"},
                    False,True
                    ],
         "case_name": "BatchMatmulv2_19",
         "expect": "success",
         "support_expect": True}

case20 = {"params": [{"shape": (3, 112, 64), "dtype": "float32", "format": "ND",
                      "ori_shape": (3, 112, 64),"ori_format": "ND"},
                    {"shape": (3, 112, 64), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 112, 64),"ori_format": "ND"},
                    {"shape": (64,), "dtype": "float32", "format": "ND", "ori_shape": (64,),"ori_format": "ND"},
                     None,
                    {"shape": (3, 64, 64), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 64, 64),"ori_format": "ND"},
                    True,False,
                    ],
         "case_name": "BatchMatmul_20",
         "expect": "success",
         "support_expect": True}

case21 = {"params": [{"shape": (3, 64, 112), "dtype": "float32", "format": "ND",
                      "ori_shape": (3, 64, 112),"ori_format": "ND"},
                    {"shape": (3, 64, 112), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 64, 112),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (3, 64, 64), "dtype": "float32", "format": "ND",
                     "ori_shape": (3, 64, 64),"ori_format": "ND"},
                    False,True,
                    ],
         "case_name": "BatchMatmul_21",
         "expect": "success",
         "support_expect": True}

# alphafold2
case22 = {"params": [{"shape": (16, 16, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 16, 32),"ori_format": "ND"},
                    {"shape": (1, 1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 32, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (16, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (16, 16, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_22",
         "expect": "success",
         "support_expect": True}

case23 = {"params": [{"shape": (16, 16, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 16, 32),"ori_format": "ND"},
                    {"shape": (1, 32, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 32, 16),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (16, 16, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 16, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_23",
         "expect": "success",
         "support_expect": True}

case24 = {"params": [{"shape": (256, 1024, 128), "dtype": "float16", "format": "ND",
                      "ori_shape": (256, 1024, 128),"ori_format": "ND"},
                    {"shape": (128, 512), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 512),"ori_format": "ND"},
                    {"shape": (512,), "dtype": "float32", "format": "ND",
                     "ori_shape": (512,),"ori_format": "ND"},
                    None,
                    {"shape": (256, 1024, 512), "dtype": "float32", "format": "ND",
                     "ori_shape": (256, 1024, 512),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_24",
         "expect": "success",
         "support_expect": True}

case25 = {"params": [{"shape": (1024, 256, 64), "dtype": "float16", "format": "ND",
                      "ori_shape": (1024, 256, 64),"ori_format": "ND"},
                    {"shape": (64, 256), "dtype": "float16", "format": "ND",
                     "ori_shape": (64, 256),"ori_format": "ND"},
                    {"shape": (256,), "dtype": "float32", "format": "ND",
                     "ori_shape": (256,),"ori_format": "ND"},
                    None,
                    {"shape": (1024, 256, 256), "dtype": "float32", "format": "ND",
                     "ori_shape": (1024, 256, 256),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_25",
         "expect": "success",
         "support_expect": True}

case_solve_bank_conflict = {"params": [{"shape": (1664, 32, 128), "dtype": "float16", "format": "ND",
                                        "ori_shape": (1664, 32, 128),"ori_format": "ND"},
                                       {"shape": (1664, 32, 128), "dtype": "float16", "format": "ND",
                                        "ori_shape": (1664, 32, 128),"ori_format": "ND"},
                                       None, None,
                                       {"shape": (1664, 32, 32), "dtype": "float32", "format": "ND",
                                        "ori_shape": (1664, 32, 32),"ori_format": "ND"},
                                       False, True,
                                      ],
                            "case_name": "BatchMatmul_solve_bank_conflict",
                            "expect": "success",
                            "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910B2"], case14)
ut_case.add_case(["Ascend910B2"], case15)
ut_case.add_case(["Ascend910A"], case16)
ut_case.add_case(["Ascend910A"], case17)
ut_case.add_case(["Ascend910A"], case18)
ut_case.add_case(["Ascend910A"], case20)
ut_case.add_case(["Ascend910A"], case21)
ut_case.add_case(["Ascend910A"], case22)
ut_case.add_case(["Ascend910A"], case23)
ut_case.add_case(["Ascend910A"], case24)
ut_case.add_case(["Ascend910A"], case25)
ut_case.add_case(["Ascend910A"], case_solve_bank_conflict)

def test_split_batch_matmul_v2(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16",
          "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16",
          "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    output_z = {"format": "FRACTAL_NZ"}
    get_op_support_info(x1, x2, output_z=output_z, trans_a=True)

def test_split_batch_matmul_v2_1(test_arg):
    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16, 32), "ori_shape": (16, 16, 32)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32), "ori_shape": (16, 32)}
    output_z = {"format": "ND"}
    get_op_support_info(x1, x2, output_z=output_z, trans_b=True)

def test_split_batch_matmul_v2_2(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16",
          "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16",
          "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    output_z = {"format": "ND"}
    get_op_support_info(x1, x2, output_z=output_z, trans_b=True)

def test_split_batch_matmul_v2_3(test_arg):
    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16, 32), "ori_shape": (16, 16, 32)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32), "ori_shape": (16, 32)}
    output_z = {"format": "FRACTAL_NZ"}
    get_op_support_info(x1, x2, output_z=output_z, trans_b=True)

def test_split_batch_matmul_trans_a(test_arg):
    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (32, 16), "ori_shape": (32, 16)}
    bias = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16), "ori_shape": (16, 16)}
    output_z = {"format": "ND"}
    get_op_support_info(x1, x2, bias=bias, output_z=output_z, trans_a=True)

ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2_1)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2_2)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2_3)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_trans_a)

def test_op_select_format(test_arg):
    from impl.batch_matmul_v2 import op_select_format
    # static shape
    op_select_format({"shape": (3, 2, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (3, 4, 5), "dtype": "float16", "format": "ND",
                      "ori_shape": (4, 5), "ori_format": "ND"},
                     )
    op_select_format({"shape": (3, 2, 4), "dtype": "float", "format": "ND",
                      "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (1, 4, 5), "dtype": "float", "format": "ND",
                      "ori_shape": (1, 4, 5), "ori_format": "ND"},
                     )
ut_case.add_cust_test_func(test_func=test_op_select_format)

case_dequant_requant_sum = (
    [{"shape": (198, 1, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
      "ori_shape": (198, 38, 29), "ori_format": "ND"},
     {"shape": (1, 12, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (29, 181), "ori_format": "ND"},
     {"shape": (181,), "dtype": "int32", "format": "ND", "ori_shape": (181,), "ori_format": "ND"},
     {"shape": (1, 12, 1, 1, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (181,), "ori_format": "NCHW"},
     {"shape": (198, 6, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
      "ori_shape": (198, 38, 181), "ori_format": "ND"},
     False,
     False,
     "requant", "batch_matmul_v2_requant_test",
     {"shape": (198, 12, 3, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
      "ori_shape": (198, 38, 181), "ori_format": "ND"}],
    [{"shape": (198, 6, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
      "ori_shape": (198, 38, 181), "ori_format": "ND"},
     {"shape": (6, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (181, 47), "ori_format": "ND"},
     {"shape": (47,), "dtype": "int32", "format": "ND", "ori_shape": (47,), "ori_format": "ND"},
     {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "NCHW"},
     {"shape": (198, 3, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
      "ori_shape": (198, 38, 47), "ori_format": "ND"},
     False,
     False,
     "dequant", "batch_matmul_v2_dequant_test",
     {"shape": (198, 3, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
      "ori_shape": (198, 38, 47), "ori_format": "ND"}],
)
def test_op_fusion_func(case):
    fusion_para = case[7]
    from impl.batch_matmul_v2 import batch_matmul_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from tbe.dsl import auto_schedule
    from te.tvm.target import cce
    from te.lang.cce import cce_build_code
    from impl.ascend_requant import ascend_requant_compute
    from te.platform.cce_conf import te_set_version

    def test_op_fusion(test_args):
        te_set_version("Ascend310P3")
        with cce():

            tensor_a = tvm.placeholder(case[0].get("shape"), name='tensor_a',
                                    attrs={'format': case[0].get("format"),
                                            "ori_shape": case[0].get("ori_shape")},
                                    dtype=case[0].get("dtype"))

            tensor_b = tvm.placeholder(case[1].get("shape"), name='tensor_b',
                                    attrs={'format': case[1].get("format"),
                                            "ori_shape": case[1].get("ori_shape")},
                                    dtype=case[1].get("dtype"))
            bias = tvm.placeholder(case[2].get("shape"), name='bias',
                                    attrs={'format': case[2].get("format"),
                                            "ori_shape": case[2].get("ori_shape")},
                                    dtype=case[2].get("dtype"))
            res = batch_matmul_compute(tensor_a, tensor_b, bias=bias, output_z=case[9], trans_a=case[5],
                                trans_b=case[6], offset_x=0, kernel_name=case[8])

            if fusion_para == "dequant":
                deq_tensor = tvm.placeholder(case[3].get("shape"), name='deq_tensor',
                                            attrs={'format': case[3].get("format"),
                                            "ori_shape": case[3].get("ori_shape")},
                                            dtype=case[3].get("dtype"),
                )
                out = ascend_dequant_compute(res, deq_tensor, case[4],
                                             sqrt_mode=False, relu_flag=False, kernel_name=case[8])

                tensor_list = [tensor_a, tensor_b, bias, deq_tensor, out]
            elif fusion_para == "requant":
                req_tensor = tvm.placeholder(case[3].get("shape"), name='deq_tensor',
                                            attrs={'format': case[3].get("format"),
                                            "ori_shape": case[3].get("ori_shape")},
                                            dtype=case[3].get("dtype"),
                )
                out = ascend_requant_compute(res, req_tensor, case[4],  kernel_name=case[8])

                tensor_list = [tensor_a, tensor_b, bias, req_tensor, out]

            sch = auto_schedule(out)
            config = {
                    "print_ir": False,
                    "need_build": True,
                    "name": case[6],
                    "tensor_list": tensor_list,
            }
            cce_build_code(sch, config)
        te_set_version("Ascend310")
    return test_op_fusion

for fusion_case in case_dequant_requant_sum:
    ut_case.add_cust_test_func(["Ascend310"], test_func=test_op_fusion_func(fusion_case))

case_batchmatmul_dequant_mul_add_fusion = [
    {"shape": (8, 25, 2, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
     "ori_shape": (8, 31, 784), "ori_format": "ND"},
    {"shape": (200, 6, 16, 32), "dtype": "int8", "format": "FRACTAL_Z",
     "ori_shape": (8, 784, 96), "ori_format": "HWCN"},
    None,
    False,
    False,
    {"shape": (8, 3, 2, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
     "ori_shape": (8, 31, 96), "ori_format": "ND"},
    "batchmatmul_v2_dequant_mul_add_fusion_test",
    {"shape": (1, 1, 1, 1, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "NCHW"},
    {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
     "ori_shape": (8, 31, 96), "ori_format": "ND"},
    "dequant",
    {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
     "ori_shape": (8, 31, 96), "ori_format": "ND"},
    {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
     "ori_shape": (8, 31, 96), "ori_format": "ND"},
    "mul",
    {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
     "ori_shape": (8, 31, 96), "ori_format": "ND"},
    {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
     "ori_shape": (8, 31, 96), "ori_format": "ND"},
    "add"]

def test_batchmatmul_dequant_mul_add_fusion_func(case):
    from impl.batch_matmul_v2 import batch_matmul_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.mul import mul_compute
    from impl.add import add_compute
    from tbe.dsl import auto_schedule
    from te.tvm.target import cce
    from te.lang.cce import cce_build_code
    from te.platform.cce_conf import te_set_version

    def test_op_fusion(test_args):
        te_set_version("Ascend310P3")
        with cce():
            tensor_a = tvm.placeholder(case[0].get("shape"), name='tensor_a',
                                    attrs={'format': case[0].get("format"),
                                            "ori_shape": case[0].get("ori_shape")},
                                    dtype=case[0].get("dtype"))

            tensor_b = tvm.placeholder(case[1].get("shape"), name='tensor_b',
                                    attrs={'format': case[1].get("format"),
                                            "ori_shape": case[1].get("ori_shape"),
                                            "ori_format": case[1].get("ori_format")},
                                    dtype=case[1].get("dtype"))

            res = batch_matmul_compute(tensor_a, tensor_b, bias=None, output_z=case[5], trans_a=case[3],
                                trans_b=case[4], offset_x=0, kernel_name=case[6])

            deq_tensor = tvm.placeholder(case[7].get("shape"), name='deq_tensor',
                                        attrs={'format': case[7].get("format"),
                                        "ori_shape": case[7].get("ori_shape")},
                                        dtype=case[7].get("dtype"))
            res = ascend_dequant_compute(res, deq_tensor, case[8], sqrt_mode=False,
                                         relu_flag=False, kernel_name=case[9])


            mul_tensor = tvm.placeholder(case[10].get("shape"), name='mul_tensor',
                                        attrs={'format': case[10].get("format"),
                                        "ori_shape": case[10].get("ori_shape")},
                                        dtype=case[10].get("dtype"),
            )
            res = mul_compute(res, mul_tensor, case[11], kernel_name=case[12])

            add_tensor = tvm.placeholder(case[13].get("shape"), name='add_tensor',
                                        attrs={'format': case[13].get("format"),
                                        "ori_shape": case[13].get("ori_shape")},
                                        dtype=case[13].get("dtype"),
            )
            out = add_compute(res, add_tensor, case[14], kernel_name=case[15])

            tensor_list = [tensor_a, tensor_b, deq_tensor, mul_tensor, add_tensor, out]
            sch = auto_schedule(out)
            config = {
                    "print_ir": False,
                    "need_build": True,
                    "name": "test_batchmatmul_dequant_mul_add_fusion_func",
                    "tensor_list": tensor_list,
            }
            cce_build_code(sch, config)
        te_set_version("Ascend310")
    return test_op_fusion


ut_case.add_cust_test_func(["Ascend310"],
                    test_func=test_batchmatmul_dequant_mul_add_fusion_func(case_batchmatmul_dequant_mul_add_fusion))


case_batch_matmul_elementwise_ub_fusion = [
    {"shape": (77, 32, 7, 16, 16), "ori_shape": (77, 100, 512),
     "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"},
    {"shape": (32, 128, 16, 16), "ori_shape": (2048, 512),
     "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"},
    None,
    None,
    {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
     "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
    False,
    True,
    0,
    "batch_matmul_elementwise_ub_fusion_test",
    {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
     "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
    {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
     "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
    "mul1",
    {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
     "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
    "sigmoid",
    {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
     "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
    "mul2"
]

def test_batch_matmul_elementwise_ub_fusion_func(case):
    from impl.batch_matmul_v2 import batch_matmul_compute
    from impl.mul import mul_compute
    from impl.sigmoid import sigmoid_compute
    from tbe.dsl import auto_schedule
    from te.tvm.target import cce
    from te.lang.cce import cce_build_code
    from te.platform.cce_conf import te_set_version

    def test_op_ub_fusion(test_args):
        te_set_version("Ascend310P3")
        with cce():
            tensor_a = tvm.placeholder(case[0].get("shape"), name='tensor_a',
                                       attrs={'format': case[0].get("format"),
                                              "ori_shape": case[0].get("ori_shape"),
                                              "ori_format": case[0].get("ori_format")},
                                       dtype=case[0].get("dtype"))
            tensor_b = tvm.placeholder(case[1].get("shape"), name='tensor_b',
                                       attrs={'format': case[1].get("format"),
                                              "ori_shape": case[1].get("ori_shape"),
                                              "ori_format": case[1].get("ori_format")},
                                       dtype=case[1].get("dtype"))
            res_bmm = batch_matmul_compute(tensor_a, tensor_b, bias=case[2], offset_w=case[3],
                                           output_z=case[4], trans_a=case[5], trans_b=case[6],
                                           offset_x=case[7], kernel_name=case[8])
            mul_tensor = tvm.placeholder(case[9].get("shape"), name='mul_tensor',
                                             attrs={'format': case[9].get("format"),
                                                    "ori_shape": case[9].get("ori_shape")},
                                             dtype=case[9].get("dtype"))
            res = mul_compute(res_bmm, mul_tensor, case[10], kernel_name=case[11])
            res = sigmoid_compute(res, case[12], kernel_name=case[13])
            out = mul_compute(res_bmm, res, case[14], kernel_name=case[15])

            tensor_list = [tensor_a, tensor_b, mul_tensor, out]
            sch = auto_schedule(out)
            config = {
                    "print_ir": False,
                    "need_build": True,
                    "name": "test_batch_matmul_elementwise_ub_fusion_func",
                    "tensor_list": tensor_list,
            }
            cce_build_code(sch, config)
        te_set_version("Ascend310")
    return test_op_ub_fusion


ut_case.add_cust_test_func(["Ascend310"],
                    test_func=test_batch_matmul_elementwise_ub_fusion_func(case_batch_matmul_elementwise_ub_fusion))


def test_op_select_format_1():
    from impl.batch_matmul_v2 import op_select_format
    # static shape
    res = op_select_format({"shape": (3, 2, 4), "dtype": "float16", "format": "ND",
                            "ori_shape": (3, 2, 4), "ori_format": "ND"},
                           {"shape": (3, 4, 5), "dtype": "float16", "format": "ND",
                            "ori_shape": (4, 5), "ori_format": "ND"}
                           )
    res_dict = json.loads(res)
    expect_res = {
        "input0": {
            "name": "x1",
            "dtype": "float16,float32,int8,int8,bfloat16,float16",
            "format": "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ"
        },
        "input1": {
            "name": "x2",
            "dtype": "float16,float32,int8,int8,bfloat16,float16",
            "format": "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_Z,FRACTAL_NZ,FRACTAL_ZN_RNN"
        },
        "input2": {
            "name": "bias",
            "dtype": "float32,float32,int32,int32,float32,float32",
            "format": "ND,ND,ND,ND,ND,ND"
        },
        "input3": {
            "name": "offset_w",
            "dtype": "int8,int8,int8,int8,int8,int8",
            "format": "ND,ND,ND,ND,ND,ND"
        },
        "output0": {
            "name": "y",
            "dtype": "float16,float32,int32,int32,bfloat16,float16",
            "format": "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ"
        },
    }
    assert expect_res == res_dict

# test mock case
def test_mock_cases(test_args):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=side_effects)):
            with patch("impl.util.platform_adapter.tbe_platform.intrinsic_check_support", MagicMock(side_effect=side_effects)):
                test_matmul_ND2ND_fp16()
                test_matmul_ND2ND_int8()
                test_matmul_ND2ND_fp32()
                test_matmul_ND2ND_fp32_1()
                test_matmul_NZ2ND_fp16()
                test_matmul_ND2NZ_fp16()
                test_matmul_NZ2NZ_fp16()
                test_matmul_NZ2NZ_int8()
                test_matmul_hf32()
                test_matmul_fixpipe_0()
                test_matmul_fixpipe_1()
                test_matmul_fixpipe_2()
                test_matmul_fixpipe_3()
                test_matmul_fixpipe_4()
                test_op_select_format_1()

ut_case.add_cust_test_func(test_func=test_mock_cases)


for case_info in nd_cases_bmm:
    ut_case.add_case(["Ascend310"], case_info)


if __name__ == '__main__':
    ut_case._case_info_map = {}
    from case_nd_in_nd_out_bmm import cases
    for case_info in cases:
        ut_case.add_case(["Ascend310"], case_info)
