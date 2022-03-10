"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Swish ut case
"""
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("Swish", None, None)


# pylint: disable=unused-argument
def calc_expect_func(input_x, output_y, scale):
    dtype = input_x["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype: %s " % dtype)
    sigmoid_value_rvec = 1 + np.exp(-scale * input_x["value"])
    res = input_x["value"] / sigmoid_value_rvec
    return res.astype(sdtype)


ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1, 2),
                                                      "dtype": "float32", "format": "ND", "ori_shape": (1, 1),
                                                      "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (1, 2),
                                                      "dtype": "float32", "format": "ND", "ori_shape": (1, 1),
                                                      "ori_format": "ND", "param_type": "output"},
                                                      1.5, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (5, 13, 4),
                                                      "dtype": "float32", "format": "ND",
                                                      "ori_shape": (5, 13, 4),
                                                      "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (5, 13, 4),
                                                      "dtype": "float32", "format": "ND",
                                                      "ori_shape": (5, 13, 4),
                                                      "ori_format": "ND", "param_type": "output"},
                                                      -2.5, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 32),
                                                      "dtype": "float16", "format": "ND",
                                                      "ori_shape": (16, 32),
                                                      "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (16, 32),
                                                      "dtype": "float16", "format": "ND",
                                                      "ori_shape": (16, 32),
                                                      "ori_format": "ND", "param_type": "output"},
                                                      1.0, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 2, 32),
                                                      "dtype": "float16", "format": "ND",
                                                      "ori_shape": (16, 2, 32),
                                                      "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (16, 2, 32),
                                                      "dtype": "float16", "format": "ND",
                                                      "ori_shape": (16, 2, 32),
                                                      "ori_format": "ND", "param_type": "output"},
                                                      1.0, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (5, 1),
                                                     "dtype": "float16", "format": "ND",
                                                     "ori_shape": (1, 1),
                                                     "ori_format": "ND", "param_type": "input"},
                                                    {"shape": (5, 1),
                                                     "dtype": "float16", "format": "ND",
                                                     "ori_shape": (1, 1), "ori_format": "ND",
                                                     "param_type": "output"}, 
                                                     1.5, ],
                                         "calc_expect_func": calc_expect_func
                                         })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (5, 5),
                                                     "dtype": "float32", "format": "ND",
                                                     "ori_shape": (1, 1),
                                                     "ori_format": "ND", "param_type": "input"},
                                                    {"shape": (5, 5),
                                                     "dtype": "float32", "format": "ND",
                                                     "ori_shape": (1, 1), "ori_format": "ND",
                                                     "param_type": "output"}, 
                                                     1.5, ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })