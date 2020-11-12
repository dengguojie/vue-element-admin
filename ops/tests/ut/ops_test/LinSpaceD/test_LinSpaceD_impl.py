#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("LinSpaceD", None, None)

case1 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "int32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"}],
         "case_name": "lin_space_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (8192, ), "dtype": "float32", "format": "NCHW", "ori_shape": (8192, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "int32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (8192, ), "dtype": "float32", "format": "NCHW", "ori_shape": (8192, ),"ori_format": "NCHW"}],
         "case_name": "lin_space_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (991, ), "dtype": "float32", "format": "NCHW", "ori_shape": (991, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "int32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (991, ), "dtype": "float32", "format": "NCHW", "ori_shape": (991, ),"ori_format": "NCHW"}],
         "case_name": "lin_space_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1024, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1, ), "dtype": "int32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                    {"shape": (1024, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW"}],
         "case_name": "lin_space_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

def calc_expect_func(input_assist, input_start, input_stop, input_num, output_op):
    x1_value = input_assist.get("value")
    x2_value = input_start.get("value")
    x3_value = input_stop.get("value")
    x4_value = input_num.get("value")

    num_divided = x4_value-1.0

    step_divider = x3_value-x2_value
    step = step_divider/num_divided

    res_temp = x1_value*step
    res = res_temp+x2_value
    return res


ut_case.add_precision_case("all", {"params": [{"shape": (1024, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "int32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1024, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW", "param_type": "output"}],
                           "calc_expect_func": calc_expect_func,
                           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {"params": [{"shape": (8192, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "int32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (8192, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW", "param_type": "output"}],
                           "calc_expect_func": calc_expect_func,
                           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {"params": [{"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "int32", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "NCHW", "ori_shape": (1024, ),"ori_format": "NCHW", "param_type": "output"}],
                           "calc_expect_func": calc_expect_func,
                           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)


