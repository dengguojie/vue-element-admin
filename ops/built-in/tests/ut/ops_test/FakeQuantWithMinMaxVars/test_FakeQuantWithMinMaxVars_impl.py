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

FakeQuantWithMinMaxVars ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("FakeQuantWithMinMaxVars", None, None)

case1 = {"params": [{"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND"}, #x
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND"},
                    8,True,
                    ],
         "case_name": "FakeQuantWithMinMaxVars_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND"}, #x
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND"},
                    8,False,
                    ],
         "case_name": "FakeQuantWithMinMaxVars_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (11, 53), "dtype": "float32", "format": "ND", "ori_shape": (11, 53),"ori_format": "ND"}, #x
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11, 53), "dtype": "float32", "format": "ND", "ori_shape": (11, 53),"ori_format": "ND"},
                    8,False,
                    ],
         "case_name": "FakeQuantWithMinMaxVars_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (11, 53, 2), "dtype": "float32", "format": "ND", "ori_shape": (11, 53, 2),"ori_format": "ND"}, #x
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11, 53, 2), "dtype": "float32", "format": "ND", "ori_shape": (11, 53, 2),"ori_format": "ND"},
                    100,False,
                    ],
         "case_name": "FakeQuantWithMinMaxVars_4",
         "expect": "failed",
         "support_expect": True}



ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)


#precision cases
def _less_compare_float32(data_x, data_y):
    min_value = 2.0 ** (-126)
    max_value = 2.0 ** (62)
    two_value = 2.0 ** (2)

    res_sub = data_y - data_x
    res_min = np.minimum(res_sub, min_value)
    res_max = np.maximum(res_min, 0)

    res_muled = res_max * max_value
    res_mul = res_muled * max_value
    res = res_mul * two_value
    return res

def _bool_both_zero_compute(juduged_min, juduged_max):

    dtype = juduged_min.dtype
    tensor_zero = juduged_min * 0.0
    min_abs = np.abs(juduged_min)
    max_abs = np.abs(juduged_max)
    min_max_replace = min_abs + max_abs

    bool_min_max_product_less_zero = _less_compare_float32(min_max_replace,
                                                           tensor_zero)
    bool_min_max_product_more_zero = _less_compare_float32(tensor_zero,
                                                           min_max_replace)

    res = bool_min_max_product_less_zero + bool_min_max_product_more_zero
    return res

def _nudged_min_max_compute(zero_point_from_min, quant_min, quant_max, scale,
                            min):

    tensor_zero = min * 0.0
    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min,
                                                      quant_min)
    bool_more_quant_max_float = _less_compare_float32(quant_max,
                                                      zero_point_from_min)

    less_quant_min_float = quant_min * bool_less_quant_min_float
    more_quant_max_float = quant_max * bool_more_quant_max_float
    tensor_one = tensor_zero + 1.0
    bool_not_less_quant_min_float = tensor_one - bool_less_quant_min_float
    bool_not_more_quant_max_float = tensor_one - bool_more_quant_max_float

    bool_between_min_max =  bool_not_less_quant_min_float * bool_not_more_quant_max_float

    between_min_max_float = zero_point_from_min * bool_between_min_max
    between_min_max_add_half_one = between_min_max_float + 0.5
    between_min_max_round = np.floor(between_min_max_add_half_one)

    nudged_zero_point_tmp = less_quant_min_float + more_quant_max_float

    nudged_zero_point = nudged_zero_point_tmp + between_min_max_round

    nudged_min_tmp = quant_min - nudged_zero_point
    nudged_max_tmp = quant_max - nudged_zero_point
    nudged_min = nudged_min_tmp * scale
    nudged_max = nudged_max_tmp * scale

    return nudged_min, nudged_max

def calc_expect_func(x, input_min, input_max, y, num_bits, narrow_range):
    quant_max = 2 ** num_bits - 1

    if not narrow_range:
        quant_min = 0
    else:
        quant_min = 1
    max = input_max['value']
    min = input_min['value']
    scale = (max-min) / (quant_max-quant_min)
    zero_point_from_min = quant_min - min/scale
    nudged_min, nudged_max = _nudged_min_max_compute(zero_point_from_min,
                                                     quant_min,
                                                     quant_max, scale, min)
    clamped_tmp = np.minimum(x['value'], nudged_max)
    clamped = np.maximum(clamped_tmp, nudged_min)
    clamped_shifted = clamped - nudged_min
    result_tmp = np.floor(clamped_shifted/scale+0.5)
    result = result_tmp*scale + nudged_min

    bool_both_zero_value = _bool_both_zero_compute(min, max)
    res = result * bool_both_zero_value
    return res

ut_case.add_precision_case(["Ascend910"], {"params": [{"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0,0.5]},
                                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0.5,1.0]},
                                              {"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND", "param_type": "output"},
                                              8,True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case(["Ascend910"], {"params": [{"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0,0.5]},
                                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0.5,1.0]},
                                              {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16),"ori_format": "ND", "param_type": "output"},
                                              8,False],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case(["Ascend910"], {"params": [{"shape": (11, 5, 2), "dtype": "float32", "format": "ND", "ori_shape": (11, 5, 2),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0,0.5]},
                                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0.5,1.0]},
                                              {"shape": (11, 5, 2), "dtype": "float32", "format": "ND", "ori_shape": (11, 5, 2),"ori_format": "ND", "param_type": "output"},
                                              8,True],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
