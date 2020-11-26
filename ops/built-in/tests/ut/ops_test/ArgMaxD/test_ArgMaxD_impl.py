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

ArgMaxD ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("ArgMaxD", None, None)

case1 = {"params": [{"shape": (5, 8,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5, 8,16,16),"ori_format": "ND"}, #x
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5, 8,16,16),"ori_format": "ND"}, #h
                    3,
                    ],
         "case_name": "ArgMaxD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND", "ori_shape": (30000, 1),"ori_format": "ND"}, #x
                    {"shape": (30000, 1), "dtype": "float32", "format": "ND", "ori_shape": (30000, 1),"ori_format": "ND"}, #h
                    1,
                    ],
         "case_name": "ArgMaxD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2,16,16), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #x
                    {"shape": (2,16,16), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #h
                    1,
                    ],
         "case_name": "ArgMaxD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2,10,1028,1,16), "dtype": "float16", "format": "ND", "ori_shape": (2,10,1028,1,16),"ori_format": "ND"}, #x
                    {"shape": (2,10,1028,1,16), "dtype": "float16", "format": "ND", "ori_shape": (2,10,1028,1,16),"ori_format": "ND"}, #h
                    4,
                    ],
         "case_name": "ArgMaxD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (2,16,16), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #x
                    {"shape": (2,16,16), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #h
                    3,
                    ],
         "case_name": "ArgMaxD_5",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

def calc_expect_func(x, y, dimension):
    x_shape = x.get("shape")
    x_value = x.get("value")
    if dimension<0:
        dimension=dimension+len(x_shape)
    output_data=np.argmax(x_value,axis = dimension)
    result = output_data.astype(np.int32)
    return (result,)

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
                                              1,
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16), "dtype": "int32", "format": "ND", "ori_shape": (2, 16),"ori_format": "ND", "param_type": "output"},
                                              2,
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

# ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 10, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 10, 16),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1, 3, 10, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 10, 16),"ori_format": "ND", "param_type": "output"},
#                                               2,
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)



