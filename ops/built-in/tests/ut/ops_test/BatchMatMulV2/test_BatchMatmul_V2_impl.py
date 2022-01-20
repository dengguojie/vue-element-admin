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
from unittest.mock import MagicMock
from unittest.mock import patch
from op_test_frame.ut import OpUT
ut_case = OpUT("BatchMatmulV2", None, None)

from tbe import tvm
from impl.batch_matmul_v2 import check_supported
from impl.batch_matmul_v2 import get_op_support_info
from test_bmmv2_mock_case import *


vals = {("CORE_NUM", ): 48,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 524288,
        ("L0C_SIZE", ): 131072,
        ("Intrinsic_fix_pipe_l0c2out",): True,
        ("Intrinsic_fix_pipe_unit_list",): True,
        ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True
        }
def side_effects(*args):
    return vals[args]

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

case15 = {"params": [{"shape": (0, 10), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (0, 10),"ori_format": "ND"},
                     {"shape": (10, 20), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (10, 20),"ori_format": "ND"},
                     None,
                     None,
                     {"shape": (0, 20), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (0, 20),"ori_format": "ND"},
                     False,False,
                     ],
          "case_name": "BatchMatmul_support_check5"}

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

case15 = {"params": [{"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 64),"ori_format": "ND"},
                     {"shape": (4, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 64, 49),"ori_format": "ND"},
                     {"shape": (49,), "dtype": "float32", "format": "ND", "ori_shape": (49,),"ori_format": "ND"},
                    None,
                    {"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 49),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_15",
         "expect": "success",
         "support_expect": True}

case16 = {"params": [{"shape": (96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (3, 64, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3,64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmulv2_16",
         "expect": "success",
         "support_expect": True}

case17 = {"params": [{"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"}, #x
                    {"shape": (64, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (64, 96),"ori_format": "NHWC"},
                    {"shape": (64,), "dtype": "float16", "format": "NHWC", "ori_shape": (64,),"ori_format": "NHWC"},
                     None,
                    {"shape": (3, 96, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (3,96, 32),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "BatchMatmulv2_17",
         "expect": "success",
         "support_expect": True}

case18 = {"params": [{"shape": (128, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 2048),"ori_format": "ND"}, #x
                    {"shape": (36, 128, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (576, 2048),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (1, 36, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 36, 16, 16),"ori_format": "FRACTAL_NZ"},
                    False,True
                    ],
         "case_name": "BatchMatmulv2_18",
         "expect": "success",
         "support_expect": True}

case19 = {"params": [{"shape": (128, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 2048),"ori_format": "ND"}, #x
                    {"shape": (36, 128, 16, 16), "dtype": "float16", "format": "FRACTAL_ZN_RNN", "ori_shape": (576, 2048),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (1, 36, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 36, 16, 16),"ori_format": "FRACTAL_NZ"},
                    False,True
                    ],
         "case_name": "BatchMatmulv2_19",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend920A"], case14)
ut_case.add_case(["Ascend920A"], case15)
ut_case.add_case(["Ascend910A"], case16)
ut_case.add_case(["Ascend910A"], case17)
ut_case.add_case(["Ascend910A"], case18)
ut_case.add_case(["Ascend910A"], case19)

def test_split_batch_matmul_v2(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    output_z = {"format": "FRACTAL_NZ"}
    get_op_support_info(x1, x2, output_z=output_z, trans_a=True)

def test_split_batch_matmul_v2_1(test_arg):
    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16, 32), "ori_shape": (16, 16, 32)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32), "ori_shape": (16, 32)}
    output_z = {"format": "ND"}
    get_op_support_info(x1, x2, output_z=output_z, trans_b=True)

def test_split_batch_matmul_v2_2(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (16, 1, 2, 16, 16), "ori_shape": (16, 32, 16)}
    output_z = {"format": "ND"}

def test_split_batch_matmul_v2_3(test_arg):
    x1 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 16, 32), "ori_shape": (16, 16, 32)}
    x2 = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (16, 32), "ori_shape": (16, 32)}
    output_z = {"format": "FRACTAL_NZ"}

ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2_1)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2_2)
ut_case.add_cust_test_func(test_func=test_split_batch_matmul_v2_3)

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
    _test_supported(case15)
    _test_supported(case19)

ut_case.add_cust_test_func(test_func=test_op_check_supported)

def test_op_select_format(test_arg):
    from impl.batch_matmul_v2 import op_select_format
    # static shape
    op_select_format({"shape": (3, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (3, 4, 5), "dtype": "float16", "format": "ND", "ori_shape": (4, 5), "ori_format": "ND"},
                     )
    op_select_format({"shape": (3, 2, 4), "dtype": "float", "format": "ND", "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (1, 4, 5), "dtype": "float", "format": "ND", "ori_shape": (1, 4, 5), "ori_format": "ND"},
                     )
ut_case.add_cust_test_func(test_func=test_op_select_format)

case_dequant_requant_sum = (
    [{"shape": (198, 1, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (198, 38, 29), "ori_format": "ND"},
     {"shape": (1, 12, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (29, 181), "ori_format": "ND"},
     {"shape": (181,), "dtype": "int32", "format": "ND", "ori_shape": (181,), "ori_format": "ND"},
     {"shape": (1, 12, 1, 1, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (181,), "ori_format": "NCHW"},
     {"shape": (198, 6, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (198, 38, 181), "ori_format": "ND"},
     False,
     False,
     "requant", "batch_matmul_v2_requant_test",
     {"shape": (198, 12, 3, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (198, 38, 181), "ori_format": "ND"}],
    [{"shape": (198, 6, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (198, 38, 181), "ori_format": "ND"},
     {"shape": (6, 3, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (181, 47), "ori_format": "ND"},
     {"shape": (47,), "dtype": "int32", "format": "ND", "ori_shape": (47,), "ori_format": "ND"},
     {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "NCHW"},
     {"shape": (198, 3, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (198, 38, 47), "ori_format": "ND"},
     False,
     False,
     "dequant", "batch_matmul_v2_dequant_test",
     {"shape": (198, 3, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (198, 38, 47), "ori_format": "ND"}], 
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
        te_set_version("Ascend710")
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
                out = ascend_dequant_compute(res, deq_tensor, case[4], sqrt_mode=False, relu_flag=False, kernel_name=case[8])

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
    
# test mock case
def test_mock_cases(test_args):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=side_effects)):
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

ut_case.add_cust_test_func(test_func=test_mock_cases)
# open this in local env
#ut_case.run(["Ascend310"])

if __name__ == '__main__':
    ut_case._case_info_map = {}
#     ut_case.add_case(["Ascend920A"], case14)
#     ut_case.add_case(["Ascend920A"], case15)

    # from case_batchmatmul_v2 import precision_cases
    # for case in precision_cases:
    #     ut_case.add_precision_case(["Ascend310", "Ascend910"], case)

    from case_nd_in_nd_out import cases
    for case in cases:
        #print(case)
        ut_case.add_case(["Ascend310"], case)
    #ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path="../../Ascend/toolkit/tools/simulator")
