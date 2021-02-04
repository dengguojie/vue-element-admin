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

AdamApplyOne ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("AdamApplyOne", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    ],
         "case_name": "AdamApplyOne_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    ],
         "case_name": "AdamApplyOne_2",
         "expect": "success",
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)

def calc_expect_func(x0, x1, x2, x3, x4,
                     mul0, mul1, mul2, mul3, add2,
                     out1, out2, out3):
    square_result = np.square(x0['value'])
    mul_3_result = square_result * mul3['value']
    mul_2_result = x1['value'] * mul2['value']
    output0 = mul_2_result + mul_3_result

    sqrt_result = np.sqrt(output0)

    add_2_result = sqrt_result + add2['value']
    mul_0_result = x2['value'] * mul0['value']
    mul_1_result = x0['value'] * mul1['value']
    output1 = mul_0_result + mul_1_result

    truediv_result = output1 / add_2_result
    mul_4_result = truediv_result * x4['value']
    output2 = x3['value'] - mul_4_result

    return output0, output1, output2

case3 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
                    ],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", case3)

def test_op_select_format(test_arg):
    from impl.adam_apply_one import op_select_format
    op_select_format({"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "output"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "output"},
                    {"shape": (3, 2, 32, 16, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 32, 16, 32, 16),"ori_format": "ND", "param_type": "output"},
                    )

ut_case.add_cust_test_func(test_func=test_op_select_format)
