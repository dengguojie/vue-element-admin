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

GeluGrad ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as NP

ut_case = OpUT("GeluGrad", None, None)

case1 = {"params": [{"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"}, #x
                    {"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    {"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    {"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"}, #x
                    {"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"}, #h
                    {"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    {"shape": (3, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"}, #x
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"}, #h
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_3",
         "expect": "success",
         "support_expect": True}
"""
case4 = {"params": [{"shape": (3, 32, 128), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"}, #x
                    {"shape": (3, 128, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 128, 32),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    ],
         "case_name": "GeluGrad_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"}, #x
                    {"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"}, #h
                    {"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"},
                    {"shape": (10.1, -1), "dtype": "float32", "format": "ND", "ori_shape": (10.1, -1),"ori_format": "ND"},
                    ],
         "case_name": "GeluGrad_5",
         "expect": RuntimeError,
         "support_expect": True}

"""
# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
#ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
#ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

def calc_expect_func(inputA, inputB, inputC, output):
    input_A_Arr = inputA['value']
    input_B_Arr = inputB['value']
    input_C_Arr = inputC['value']
    f4 = (input_B_Arr + NP.power(input_B_Arr, 3)*0.044715)*0.7978846
    res = input_C_Arr
    res1 = 0.7978846*(1.0 + 0.134145*NP.power(input_B_Arr, 2))
    tanh_f4 = NP.tanh(f4)
    res2 = input_B_Arr*0.5*(1.0- tanh_f4*tanh_f4)*res1
    # outputArr1 = res/input_B_Arr + res2
    outputArr1 = (tanh_f4+1)*0.5 + res2
    outputArr = outputArr1 * input_A_Arr
    return outputArr

precision_case1 = {"params": [{"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case2 = {"params": [{"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case3 = {"params": [{"shape": (1,2), "dtype": "float32", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (1,2), "dtype": "float32", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (1,2), "dtype": "float32", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (1,2), "dtype": "float32", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case4 = {"params": [{"shape": (512,256), "dtype": "float32", "format": "ND", "ori_shape": (512,256),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (512,256), "dtype": "float32", "format": "ND", "ori_shape": (512,256),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (512,256), "dtype": "float32", "format": "ND", "ori_shape": (512,256),"ori_format": "ND","param_type":"input", "value_range":[-1,1]},
                              {"shape": (512,256), "dtype": "float32", "format": "ND", "ori_shape": (512,256),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}

ut_case.add_precision_case("Ascend910",precision_case1)
ut_case.add_precision_case("Ascend910",precision_case2)
ut_case.add_precision_case("Ascend910",precision_case3)
ut_case.add_precision_case("Ascend910",precision_case4)


if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")
