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

FakeQuantWithMinMaxArgsGradient ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("FakeQuantWithMinMaxArgsGradient", None, None)

case1 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "NHWC", "ori_shape": (128, 255, 36),"ori_format": "NHWC"}, #x
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NHWC", "ori_shape": (128, 255, 36),"ori_format": "NHWC"},
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NHWC", "ori_shape": (128, 255, 36),"ori_format": "NHWC"},
                    ],
         "case_name": "FakeQuantWithMinMaxArgsGradient_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1024, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (1024, 1024),"ori_format": "NHWC"}, #x
                    {"shape": (1024, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (1024, 1024),"ori_format": "NHWC"},
                    {"shape": (1024, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (1024, 1024),"ori_format": "NHWC"},
                    -10.67,-5.55,6
                    ],
         "case_name": "FakeQuantWithMinMaxArgsGradient_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (123,), "dtype": "float32", "format": "NCHW", "ori_shape": (123,),"ori_format": "NCHW"}, #x
                    {"shape": (123,), "dtype": "float32", "format": "NCHW", "ori_shape": (123,),"ori_format": "NCHW"},
                    {"shape": (123,), "dtype": "float32", "format": "NCHW", "ori_shape": (123,),"ori_format": "NCHW"},
                    7.778,30.123,9,True
                    ],
         "case_name": "FakeQuantWithMinMaxArgsGradient_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (99991,), "dtype": "float32", "format": "NCHW", "ori_shape": (99991,),"ori_format": "NCHW"}, #x
                    {"shape": (99991,), "dtype": "float32", "format": "NCHW", "ori_shape": (99991,),"ori_format": "NCHW"},
                    {"shape": (99991,), "dtype": "float32", "format": "NCHW", "ori_shape": (99991,),"ori_format": "NCHW"},
                    -7,8,4,False
                    ],
         "case_name": "FakeQuantWithMinMaxArgsGradient_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"}, #x
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"},
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"},
                    -10.67,-5.55,6,True
                    ],
         "case_name": "FakeQuantWithMinMaxArgsGradient_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"}, #x
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"},
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"},
                    -5,-5,6,True
                    ],
         "case_name": "FakeQuantWithMinMaxArgsGradient_6",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)

#precision cases
def _nudge_min_max_gradient(min, max, num_bits, narrow_range):
    quant_max = (2 ** num_bits) - 1

    if narrow_range is False:
        quant_min = 0.00
    else:
        quant_min = 1.00

    scale = (max - min) / (float(quant_max) - quant_min)

    zeor_point_from_min = quant_min - min / scale

    # Calculate the maximum and minimum values of the quantization
    if zeor_point_from_min < quant_min:
        nudged_zero_point = quant_min
    elif zeor_point_from_min > quant_max:
        nudged_zero_point = quant_max
    else:
        nudged_zero_point = (zeor_point_from_min + 0.5) // 1

    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale

    return nudged_min, nudged_max

def _cmpare_value(x, nudged_min, nudged_max):

    min_value = 2.0**(-126)
    max_value = 2.0**(62)
    max_value_one = 2.0**(2)

    nudged_min_neg = nudged_min * (-1.0)
    nudged_max_neg = nudged_max * (-1.0)
    vmax_tmp = np.maximum(x + nudged_min_neg + min_value, 0)
    vmin_tmp = np.maximum((x + nudged_max_neg)*(-1.0) + min_value, 0)

    one_tmp = vmax_tmp * vmin_tmp
    one_min = np.minimum(one_tmp, min_value)
    between_nudged_min_max = one_min*max_value*max_value*max_value_one

    return between_nudged_min_max

def calc_expect_func(gradients, x, y, min, max, num_bits, narrow_range):
    nudged_min, nudged_max = _nudge_min_max_gradient(min, max, num_bits,
                                                     narrow_range)
    between_nudged_min_max = _cmpare_value(x['value'], nudged_min, nudged_max)

    res = gradients['value'] * between_nudged_min_max
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              7.778,30.123,9,True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (123, ), "dtype": "float32", "format": "ND", "ori_shape": (123, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (123, ), "dtype": "float32", "format": "ND", "ori_shape": (123, ),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (123, ), "dtype": "float32", "format": "ND", "ori_shape": (123, ),"ori_format": "ND", "param_type": "output"},
                                              7.778,30.123,9,True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 16),"ori_format": "ND", "param_type": "output"},
                                              7.778,30.123,9,True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
