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

TruncateMod ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("TruncateMod", None, None)

case1 = {"params": [{"shape": (10,10241), "dtype": "float16", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"}, #x
                    {"shape": (10,10241), "dtype": "float16", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    {"shape": (10,10241), "dtype": "float16", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    ],
         "case_name": "TruncateMod_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 8), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"}, #x
                    {"shape": (1, 8), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
                    {"shape": (1, 8), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
                    ],
         "case_name": "TruncateMod_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1, 1, 3), "dtype": "int32", "format": "ND", "ori_shape": (1, 1, 3),"ori_format": "ND"}, #x
                    {"shape": (1, 1, 3), "dtype": "int32", "format": "ND", "ori_shape": (1, 1, 3),"ori_format": "ND"},
                    {"shape": (1, 1, 3), "dtype": "int32", "format": "ND", "ori_shape": (1, 1, 3),"ori_format": "ND"},
                    ],
         "case_name": "TruncateMod_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),"ori_format": "ND"},
                    {"shape": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),"ori_format": "ND"},
                    ],
         "case_name": "TruncateMod_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (1, 8), "dtype": "float64", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"}, #x
                    {"shape": (1, 8), "dtype": "float64", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
                    {"shape": (1, 8), "dtype": "float64", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
                    ],
         "case_name": "TruncateMod_5",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

def calc_expect_func(x1, x2, y):
    x1_value = x1.get("value")
    x2_value = x2.get("value")

    res = np.fmod(x1_value, x2_value)
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

#ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
#                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
#                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "output"},
#                                              ],
#                                   "calc_expect_func": calc_expect_func,
#                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)



