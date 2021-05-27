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

fills ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
import os

ut_case = OpUT("Fills")

case1 = {"params": [{"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND"},
                    {"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND"},
                    3.0],
         "case_name": "Fills_1",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case2 = {"params": [{"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND"},
                    {"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND"},
                    3.0],
         "case_name": "Fills_2",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case3 = {"params": [{"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND"},
                    {"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND"},
                    3.0],
         "case_name": "Fills_3",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

case4 = {"params": [{"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND"},
                    {"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND"},
                    3.0],
         "case_name": "Fills_4",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}

ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
ut_case.add_case(["Ascend910"], case4)


def calc_expect_func(input_x, output, hg):
    shape_x=input_x.get("shape")
    input_y=np.ones((1,)).astype(input_x["dtype"])
    input_y=input_y*hg
    output_arr=input_x.get("value")+input_y
    output=np.ones(input_x["shape"])*hg
    return output

ut_case.add_precision_case("all", {
    "params": [{"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND", "param_type": "input"},
               {"shape": (125, 125), "dtype": "float16", "format": "ND", "ori_shape": (125, 125),"ori_format": "ND", "param_type": "output"},
               3.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {"params": [{"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 30, 100), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100),"ori_format": "ND", "param_type": "output"},
                    3.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {"params": [{"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 30, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16),"ori_format": "ND", "param_type": "output"},
                    3.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {"params": [{"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND", "param_type": "input"},
                    {"shape": (3, 30, 100, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 100, 16, 17),"ori_format": "ND", "param_type": "output"},
                    3.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (3, 30, 10, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 10, 16, 17),"ori_format": "ND", "param_type": "input"},
               {"shape": (3, 30, 10, 16, 17), "dtype": "float16", "format": "ND", "ori_shape": (3, 30, 10, 16, 17),"ori_format": "ND", "param_type": "output"},
               3.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
