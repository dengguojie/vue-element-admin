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

AllClose ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("AllClose", None, None)

case1 = {"params": [{"shape": (5, 8,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5, 8,16,16),"ori_format": "ND"},
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "ND", "ori_shape": (5, 8,16,16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001,0.0001,
                    ],
         "case_name": "AllClose_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND", "ori_shape": (30000, 1),"ori_format": "ND"}, #x
                    {"shape": (30000, 1), "dtype": "float32", "format": "ND", "ori_shape": (30000, 1),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001,0.0001,
                    ],
         "case_name": "AllClose_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2,16,32), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #x
                    {"shape": (2,16,32), "dtype": "float16", "format": "ND", "ori_shape": (2,16,16),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001,0.0001,
                    ],
         "case_name": "AllClose_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2,10,1028,1,16), "dtype": "float16", "format": "ND", "ori_shape": (2,10,1028,1,16),"ori_format": "ND"}, #x
                    {"shape": (2,10,1028,1,16), "dtype": "float16", "format": "ND", "ori_shape": (2,10,1028,1,16),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.001,0.001,
                    ],
         "case_name": "AllClose_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)

# def calc_expect_func(x, y, num, diff, atol, rtol):
#     x_value = x.get("value")
#     y_value = y.get("value")

#     sub = np.abs(np.subtract(x_value, y_value))
#     res_dif = atol + rtol * np.abs(y_value)
#     res_cmp = np.greater(res_dif, sub)
#     res = np.where(res_cmp, 0, 1)
#     num = np.sum(res).astype(np.int32)

#     div = sub / np.abs(y_value) * res
#     diff = np.max(div).astype(np.float16)
#     return (num, diff)

# ut_case.add_precision_case("all", {"params": [{"shape": (1, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
#                                               {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
#                                               0.001, 0.001,
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })

# ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "float32", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (2, 16, 32), "dtype": "float32", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
#                                               {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"},
#                                               0.0001,0.0001,
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
#                                    })


# # ============ auto gen ["Ascend910"] test cases end =================

# if __name__ == '__main__':
#     user_home_path = os.path.expanduser("~")
#     simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
#     ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)



