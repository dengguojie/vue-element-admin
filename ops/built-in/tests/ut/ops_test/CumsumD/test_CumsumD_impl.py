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

CumsumD ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("CumsumD", "impl.cumsum_d", "cumsum_d")

def calc_expect_func(x, y, axis, exclusive, reverse):
    shape = x['shape']
    input_x = x['value']
    dtype = x['dtype']
    if axis < 0:
        axis = len(shape) + axis
    each = 1
    for i in range(axis+1, len(shape)):
        each = each*shape[i]

    each_loop = shape[axis]

    outer_loop = 1
    for i in range(0, axis):
        outer_loop = outer_loop * shape[i]

    ret = np.zeros(shape=(outer_loop, each_loop, each)).astype(dtype)
    input_x = input_x.reshape(outer_loop, each_loop, each)

    if not exclusive and not reverse:
        for o in range(0, outer_loop):
            ret[o, 0, :] = input_x[o, 0, :]
            for e in range(1, each_loop):
                ret[o, e, :] = input_x[o, e, :] + ret[o, e-1, :]

    if exclusive and not reverse:
        for o in range(0, outer_loop):
            ret[o, 0, :] = 0
            for e in range(1, each_loop):
                ret[o, e, :] = input_x[o, e-1, :] + ret[o, e-1, :]

    if not exclusive and reverse:
        for o in range(0, outer_loop):
            ret[o, each_loop-1, :] = input_x[o, each_loop-1, :]
            for e in range(1, each_loop):
                ret[o, each_loop-1-e, :] = ret[o, each_loop-e,:] + input_x[o, each_loop-1-e, :]

    if exclusive and reverse:
        for o in range(0, outer_loop):
            ret[o, each_loop-1, :] = 0
            for e in range(1, each_loop):
                ret[o, each_loop-1-e, :] = ret[o, each_loop-e,:] + input_x[o, each_loop-e, :]

    ret = ret.reshape(y['shape'])
    return ret

def test_get_op_support_info(test_arg):
    from impl.cumsum_d import get_op_support_info
    get_op_support_info({"shape": (15, 80, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"}, 
                        {"shape": (15, 80, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"})

def test_check_support(test_arg):
    from impl.cumsum_d import check_supported
    check_supported({"shape": (15, 80, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"}, 
                    {"shape": (15, 80, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"},
                    axis=-1)

case1 = {"params": [{"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"}, #x
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    ],
         "case_name": "CumsumD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 2), "dtype": "int8", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"}, #x
                    {"shape": (1, 2), "dtype": "int8", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    True,
],
"case_name": "CumsumD_2",
"expect": "success",
"support_expect": True}

case3 = {"params": [{"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"}, #x
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    ],
         "case_name": "CumsumD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1, 3), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"}, #x
                    {"shape": (1, 3), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    ],
         "case_name": "CumsumD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (15, 80, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"}, #x
                    {"shape": (15, 80, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"},
                    2, True
],
"case_name": "CumsumD_5",
"expect": "success",
"support_expect": True}

case6 = {"params": [{"shape": (15, 80, 2, 76), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 76),"ori_format": "ND"}, #x
                    {"shape": (15, 80, 2, 76), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 76),"ori_format": "ND"},
                    -1, True, True,
],
"case_name": "CumsumD_6",
"expect": "success",
"support_expect": True}

case7 = {"params": [{"shape": (15, 8, 50, 272), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 272),"ori_format": "ND"}, #x
                    {"shape": (15, 8, 50, 272), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 272),"ori_format": "ND"},
                    1, True, False,
],
"case_name": "CumsumD_7",
"expect": "success",
"support_expect": True}

case8 = {"params": [{"shape": (15, 8, 50, 271), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 271),"ori_format": "ND"}, #x
                    {"shape": (15, 8, 50, 271), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 271),"ori_format": "ND"},
                    1, False, False,
],
"case_name": "CumsumD_8",
"expect": "success",
"support_expect": True}

case9 = {"params": [{"shape": (15, 8, 50, 270), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 270),"ori_format": "ND"}, #x
                    {"shape": (15, 8, 50, 270), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 270),"ori_format": "ND"},
                    1, False, True,
],
"case_name": "CumsumD_9",
"expect": "success",
"support_expect": True}

case10 = {"params": [{"shape": (3, 8, 50, 273), "dtype": "float32", "format": "ND", "ori_shape": (3, 8, 50, 273),"ori_format": "ND"}, #x
                     {"shape": (15, 8, 50, 273), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 273),"ori_format": "ND"},
                     1, True, True,
],
"case_name": "CumsumD_10",
"expect": "success",
"support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case8)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case9)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case10)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)
ut_case.add_cust_test_func(test_func=test_check_support)

precision_case1 = {"params": [{"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND","param_type":"input"}, #x
                              {"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND","param_type":"output"},
                              0, False, False
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (11, 33), "dtype": "float16", "format": "ND", "ori_shape": (11, 33), "ori_format": "ND","param_type":"input"}, #x
                              {"shape": (11, 33), "dtype": "float16", "format": "ND", "ori_shape": (11, 33), "ori_format": "ND","param_type":"output"},
                              1, True, False
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (2, 3, 4, 5, 6, 7), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 5, 6, 7), "ori_format": "ND","param_type":"input"}, #x
                              {"shape": (2, 3, 4, 5, 6, 7), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 5, 6, 7), "ori_format": "ND","param_type":"output"},
                              0, False, False
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case4 = {"params": [{"shape": (12, 5), "dtype": "float16", "format": "ND", "ori_shape": (12, 5), "ori_format": "ND","param_type":"input"}, #x
                              {"shape": (12, 5), "dtype": "float16", "format": "ND", "ori_shape": (12, 5), "ori_format": "ND","param_type":"output"},
                              0, False, False
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case5 = {"params": [{"shape": (250, 3), "dtype": "float16", "format": "ND", "ori_shape": (250, 3), "ori_format": "ND","param_type":"input"}, #x
                              {"shape": (250, 3), "dtype": "float16", "format": "ND", "ori_shape": (250, 3), "ori_format": "ND","param_type":"output"},
                              0, False, False
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)
ut_case.add_precision_case("Ascend910", precision_case4)
ut_case.add_precision_case("Ascend910", precision_case5)


