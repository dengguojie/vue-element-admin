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

Hardswish ut case
"""
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("Hardswish", None, None)

# pylint: disable=unused-argument
def calc_expect_func(input_x, output_y):
    dtype = input_x["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype: %s " % dtype)
    res = input_x["value"] * (input_x["value"] + 3) / 6
    res[res <= -3] = 0
    res[res >= 3] = input_x["value"][res >= 3]
    return res.astype(sdtype)


ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1, 2),
                                                      "dtype": "float32", "format": "ND", "ori_shape": (1, 1),
                                                      "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND", "param_type": "output"}, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4), "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4), "ori_format": "ND", "param_type": "output"}, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"}, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32), "ori_format": "ND", "param_type": "input"},
                                                     {"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32), "ori_format": "ND", "param_type": "output"}, ],
                                          "calc_expect_func": calc_expect_func
                                          })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND", "param_type": "output"}, ],
                                         "calc_expect_func": calc_expect_func
                                         })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4), "ori_format": "ND", "param_type": "input"},
                                                    {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4), "ori_format": "ND", "param_type": "output"}, ],
                                         "calc_expect_func": calc_expect_func
                                         })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32), "ori_format": "ND", "param_type": "output"}, ],
                                         "calc_expect_func": calc_expect_func
                                         })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32), "ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32), "ori_format": "ND", "param_type": "output"}, ],
                                         "calc_expect_func": calc_expect_func
                                         })
