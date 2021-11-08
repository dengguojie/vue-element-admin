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

LambUpdateWithLrV2 ut case
"""
from op_test_frame.ut import OpUT
import os
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("LambUpdateWithLrV2", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #x
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #h
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #c
                    {"shape": (512, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (512, 1024),"ori_format": "NCHW"}, #w
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},  #b
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #mask
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #ft
                    {"shape": (512, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (512, 1024),"ori_format": "NCHW"}, #ot
                    ],
         "case_name": "LambUpdateWithLrV2_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #x
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #h
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #c
                    {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"}, #w
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},  #b
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #mask
                    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}, #ft
                    {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"}, #ot
                    ],
         "case_name": "LambUpdateWithLrV2_2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)


def calc_expect_func(input0, input1, input2, input3, input4, greater_y, select_e, res):
    x1_value = input0.get("value")
    x2_value = input1.get("value")
    x3_value = input2.get("value")
    x4_value = input3.get("value")
    x5_value = input4.get("value")
    x6_value = greater_y.get("value")
    x7_value = select_e.get("value")

    greater_0 = np.greater(x1_value, x6_value)

    greater_1 = np.greater(x2_value, x6_value)
    truediv0 = np.true_divide(x1_value, x2_value)

    select_0 = np.select([greater_1, ~greater_1], [truediv0, x7_value])
    select_1 = np.select([greater_0, ~greater_0], [select_0, x7_value])

    mul_5 = np.multiply(x3_value, select_1)
    mul_6 = np.multiply(mul_5, x4_value)

    res = np.subtract(x5_value, mul_6)
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #x
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #h
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #c
                                              {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW", "param_type": "input"}, #w
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"},  #b
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #mask
                                              {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #ft
                                              {"shape": (1024,), "dtype": "float32", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW", "param_type": "output"}, #ot
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #x
                                              {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #h
                                              {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #c
                                              {"shape": (512, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (512, 1024),"ori_format": "NCHW", "param_type": "input"}, #w
                                              {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"},  #b
                                              {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #mask
                                              {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW", "param_type": "input"}, #ft
                                              {"shape": (512, 1024), "dtype": "float16", "format": "NCHW", "ori_shape": (512, 1024),"ori_format": "NCHW", "param_type": "output"}, #ot
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
