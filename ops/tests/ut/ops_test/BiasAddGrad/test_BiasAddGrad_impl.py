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

BiasAddGrad ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("BiasAddGrad", None, None)

case1 = {"params": [{"shape": (2,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,3),"ori_format": "NHWC"}, #x
                    {"shape": (2,3), "dtype": "float16", "format": "NHWC", "ori_shape":(2,3),"ori_format": "NHWC"},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (77, 30522), "dtype": "float32", "format": "NHWC", "ori_shape": (77, 30522),"ori_format": "NHWC"}, #x
                    {"shape": (77, 30522), "dtype": "float32", "format": "NHWC", "ori_shape": (77, 30522),"ori_format": "NHWC"},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (10, 10, 10, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (10, 10, 10, 10),"ori_format": "NCHW"}, #x
                    {"shape": (10, 10, 10, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (10, 10, 10, 10),"ori_format": "NCHW"},
                    "NCHW"
                    ],
         "case_name": "BiasAddGrad_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)

def calc_expect_func(input, output, data_format):
    inputArr = input['value'].astype('float32')
    input_shape = input['shape']
    if data_format == "NHWC":
        axis = [x for x in range(len(input_shape)-1)][::-1]
        outputArr = inputArr
        for i in axis:
            outputArr = np.sum(outputArr, i)
    else:
        outputArr = np.sum(np.sum(np.sum(inputArr, 3), 2), 0)
    outputArr = outputArr.astype(output['dtype'])
    return outputArr

precision_case1 = {"params": [{"shape": (10, 20), "dtype": "float16", "format": "ND", "ori_shape": (10, 20),"ori_format": "ND","param_type":"input"},
                              {"shape": (20,), "dtype": "float16", "format": "ND", "ori_shape": (20,),"ori_format": "ND","param_type":"output"}, "NHWC"],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (100,100), "dtype": "float16", "format": "ND", "ori_shape": (100,100),"ori_format": "ND","param_type":"input"},
                              {"shape": (100,), "dtype": "float16", "format": "ND", "ori_shape": (100,),"ori_format": "ND","param_type":"output"}, "NHWC"],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input"},
                              {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND","param_type":"output"}, "NHWC"],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)


if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")
