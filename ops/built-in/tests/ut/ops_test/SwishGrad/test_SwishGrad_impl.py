"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

SwishGrad ut case
"""
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("SwishGrad", None, None)


case1 = {"params": [{"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    {"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    {"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    {"shape": (5, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (5, 10),"ori_format": "NHWC"},
                    1.5],
         "case_name": "SwishGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    {"shape": (3, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    {"shape": (3, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    {"shape": (3, 96), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 96),"ori_format": "NHWC"},
                    -1.5],
         "case_name": "SwishGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    {"shape": (3, 24, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 24, 96),"ori_format": "NHWC"},
                    ],
         "case_name": "SwishGrad_3",
         "expect": "success",
         "support_expect": True}


# ============ auto gen ["Ascend910"] test cases end =================
def calc_expect_func(input_x, output_x, output_y, output_z, beta):
    grad = output_y["value"] / output_x["value"] \
           * (1 - beta * output_y["value"]) + beta * output_y["value"]
    res = (grad * input_x["value"]).astype(output_z["dtype"])
    return res


ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (1, ), "shape": (1, ), "param_type": "output"},
                1.5],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (11, 33), "shape": (11, 33), "param_type": "output"},
                1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 32), "shape": (16, 2, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 32), "shape": (16, 2, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 32), "shape": (16, 2, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 32), "shape": (16, 2, 32), "param_type": "output"},
                1.5],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "output"},
                1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
