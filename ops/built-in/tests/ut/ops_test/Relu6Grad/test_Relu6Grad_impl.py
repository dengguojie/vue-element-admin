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

Relu6Grad ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("Relu6Grad", None, None)

case1 = {"params": [{"shape": (16, 1024), "dtype": "float32", "format": "ND", "ori_shape": (16, 1024),"ori_format": "ND"},
                    {"shape": (16, 1024), "dtype": "float32", "format": "ND", "ori_shape": (16, 1024),"ori_format": "ND"},
                    {"shape": (16, 1024), "dtype": "float32", "format": "ND", "ori_shape": (16, 1024),"ori_format": "ND"}],
         "case_name": "relu6_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (32, 112, 112, 32), "dtype": "float16", "format": "ND", "ori_shape": (32, 112, 112, 32),"ori_format": "ND"},
                    {"shape": (32, 112, 112, 32), "dtype": "float16", "format": "ND", "ori_shape": (32, 112, 112, 32),"ori_format": "ND"},
                    {"shape": (32, 112, 112, 32), "dtype": "float16", "format": "ND", "ori_shape": (32, 112, 112, 32),"ori_format": "ND"}
                    ],
         "case_name": "relu6_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (32, 112, 112, 96), "dtype": "float32", "format": "ND", "ori_shape": (32, 112, 112, 96),"ori_format": "ND"},
                    {"shape": (32, 112, 112, 96), "dtype": "float32", "format": "ND", "ori_shape": (32, 112, 112, 96),"ori_format": "ND"},
                    {"shape": (32, 112, 112, 96), "dtype": "float32", "format": "ND", "ori_shape": (32, 112, 112, 96),"ori_format": "ND"}
                    ],
         "case_name": "relu6_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (32, 56, 56, 96), "dtype": "float32", "format": "ND", "ori_shape": (32, 56, 56, 96),"ori_format": "ND"},
                    {"shape": (32, 56, 56, 96), "dtype": "float32", "format": "ND", "ori_shape": (32, 56, 56, 96),"ori_format": "ND"},
                    {"shape": (32, 56, 56, 96), "dtype": "float32", "format": "ND", "ori_shape": (32, 56, 56, 96),"ori_format": "ND"}
                    ],
         "case_name": "relu6_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (32, 28, 28, 192), "dtype": "float32", "format": "ND", "ori_shape": (32, 28, 28, 192),"ori_format": "ND"},
                    {"shape": (32, 28, 28, 192), "dtype": "float32", "format": "ND", "ori_shape": (32, 28, 28, 192),"ori_format": "ND"},
                    {"shape": (32, 28, 28, 192), "dtype": "float32", "format": "ND", "ori_shape": (32, 28, 28, 192),"ori_format": "ND"}
                    ],
         "case_name": "relu6_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (32, 14, 14, 192), "dtype": "float32", "format": "ND", "ori_shape": (32, 14, 14, 192),"ori_format": "ND"},
                    {"shape": (32, 14, 14, 192), "dtype": "float32", "format": "ND", "ori_shape": (32, 14, 14, 192),"ori_format": "ND"},
                    {"shape": (32, 14, 14, 192), "dtype": "float32", "format": "ND", "ori_shape": (32, 14, 14, 192),"ori_format": "ND"}
                    ],
         "case_name": "relu6_grad_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (32, 14, 14, 384), "dtype": "float32", "format": "ND", "ori_shape": (32, 14, 14, 384),"ori_format": "ND"},
                    {"shape": (32, 14, 14, 384), "dtype": "float32", "format": "ND", "ori_shape": (32, 14, 14, 384),"ori_format": "ND"},
                    {"shape": (32, 14, 14, 384), "dtype": "float32", "format": "ND", "ori_shape": (32, 14, 14, 384),"ori_format": "ND"}
                    ],
         "case_name": "relu6_grad_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}



ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)


def calc_expect_func(input_grad, input_x, output_y):
    dtype = input_grad["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    cond_lt_6 = input_x["value"] < 6
    cond_gt_0 = input_x["value"] > 0

    tmpResult = np.where(cond_gt_0, input_grad["value"], 0)
    return np.where(cond_lt_6, tmpResult, 0).astype(sdtype)

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 1024), "shape": (16, 1024), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 1024), "shape": (16, 1024), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 1024), "shape": (16, 1024), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 56, 56), "shape": (32, 56, 56), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 56, 56), "shape": (32, 56, 56), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 56, 56), "shape": (32, 56, 56), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 14, 14, 384), "shape": (1, 14, 14, 384), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 14, 14, 384), "shape": (1, 14, 14, 384), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 14, 14, 384), "shape": (1, 14, 14, 384), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 14, 14, 7), "shape": (1, 14, 14, 7), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 14, 14, 7), "shape": (1, 14, 14, 7), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 14, 14, 7), "shape": (1, 14, 14, 7), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run()
    exit(0)