#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("ReluGradV2", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"}],
         "case_name": "relu_grad_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,8), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,8),"ori_format": "NC1HWC0"}],
         "case_name": "relu_grad_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "NC1HWC0"},
                    {"shape": (1,), "dtype": "uint8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,2,8), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,8),"ori_format": "NC1HWC0"}],
         "case_name": "relu_grad_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)


def calc_expect_func(x1, x2, y):

    input1_data = x1.get("value")
    input2_data = x2.get("value")

    result = np.where(input1_data, input2_data, 0)

    return result


ut_case.add_precision_case("all", {"params": [{"shape": (1, ), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "uint8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "uint8", "format": "NC1HWC0", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, ), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "uint8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "uint8", "format": "NC1HWC0", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, ), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "uint8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "uint8", "format": "NC1HWC0", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

