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

DecodeCornerpointsTargetBg ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("DecodeCornerpointsTargetBg", None, None)

case1 = {"params": [{"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"}, #x
                    {"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"},
                    {"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"},
                    ],
         "case_name": "DecodeCornerpointsTargetBg_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (65400,4), "dtype": "float16", "format": "ND", "ori_shape": (65400,4),"ori_format": "ND"}, #x
                    {"shape": (65400,4), "dtype": "float16", "format": "ND", "ori_shape": (65400,4),"ori_format": "ND"},
                    {"shape": (65400,4), "dtype": "float16", "format": "ND", "ori_shape": (65400,4),"ori_format": "ND"},
                    ],
         "case_name": "DecodeCornerpointsTargetBg_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (4,16,4), "dtype": "float16", "format": "ND", "ori_shape": (4,16,4),"ori_format": "ND"}, #x
                    {"shape": (4,16,4), "dtype": "float16", "format": "ND", "ori_shape": (4,16,4),"ori_format": "ND"},
                    {"shape": (4,16,4), "dtype": "float16", "format": "ND", "ori_shape": (4,16,4),"ori_format": "ND"},
                    ],
         "case_name": "DecodeCornerpointsTargetBg_3",
         "expect": RuntimeError,
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310"], case1)
ut_case.add_case(["Ascend910A","Ascend310"], case2)
ut_case.add_case(["Ascend910A","Ascend310"], case3)

def calc_expect_func(inputA, inputB, output):
    input_shape_x = inputA['shape']
    data_a = inputA['value']
    data_b = inputB['value']
    a_x1y1x2y2 = np.reshape(data_a, (-1, 2, 2))
    a_x1y1, a_x2y2 = np.split(a_x1y1x2y2, 2, axis=1)
    b_x1y1x2y2 = np.reshape(data_b, (-1, 2, 2))
    b_x1y1, b_x2y2 = np.split(b_x1y1x2y2, 2, axis=1)
    waha = b_x2y2 - b_x1y1
    xaya = (b_x2y2 + b_x1y1) * 0.5
    c_x1y1 = a_x1y1 * waha + xaya
    c_x2y2 = a_x2y2 * waha + xaya
    c = np.concatenate([c_x1y1, c_x2y2], axis=1)
    data_c = np.reshape(c, input_shape_x)
    return data_c

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 4), "dtype": "float16", "format": "ND", "ori_shape": (16, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 4), "dtype": "float16", "format": "ND", "ori_shape": (16, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 4), "dtype": "float16", "format": "ND", "ori_shape": (16, 4),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (48, 4), "dtype": "float16", "format": "ND", "ori_shape": (48, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (48, 4), "dtype": "float16", "format": "ND", "ori_shape": (48, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (48, 4), "dtype": "float16", "format": "ND", "ori_shape": (48, 4),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (32, 4), "dtype": "float16", "format": "ND", "ori_shape": (32, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 4), "dtype": "float16", "format": "ND", "ori_shape": (32, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32, 4), "dtype": "float16", "format": "ND", "ori_shape": (32, 4),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2, 4), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2, 4), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2, 4), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })

