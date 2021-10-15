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

FakeQuantWithMinMaxVarsGradient ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("FakeQuantWithMinMaxVarsGradient", None, None)

case1 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"}, #x
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    16,True,
                    ],
         "case_name": "FakeQuantWithMinMaxVarsGradient_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"}, #x
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    6,True,
                    ],
         "case_name": "FakeQuantWithMinMaxVarsGradient_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (7, 6, 1), "dtype": "float32", "format": "ND", "ori_shape": (7, 6, 1),"ori_format": "ND"}, #x
                    {"shape": (7, 6, 1), "dtype": "float32", "format": "ND", "ori_shape": (7, 6, 1),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (7, 6, 1), "dtype": "float32", "format": "ND", "ori_shape": (7, 6, 1),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    6,False,
                    ],
         "case_name": "FakeQuantWithMinMaxVarsGradient_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (7, 6, 1), "dtype": "float32", "format": "ND", "ori_shape": (7, 6, 1),"ori_format": "ND"}, #x
                    {"shape": (7, 6, 1), "dtype": "float32", "format": "ND", "ori_shape": (7, 6, 1),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (7, 6, 1), "dtype": "float32", "format": "ND", "ori_shape": (7, 6, 1),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    1,False,
                    ],
         "case_name": "FakeQuantWithMinMaxVarsGradient_4",
         "expect": RuntimeError,
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend710"], case1)
#ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend710"], case4)

#precision cases
#model value wrong
def _bool_negate(input_bool):
    tensor_one = np.ones(input_bool.shape)
    output_bool = tensor_one - input_bool
    return output_bool

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

def _both_min_max_zero(input_min, input_max, input_shape, dtype):
    min_broad = np.broadcast_to(input_min, input_shape)
    max_broad = np.broadcast_to(input_max, input_shape)
    tensor_zero = min_broad * 0
    min_abs = np.abs(min_broad)
    max_abs = np.abs(max_broad)
    min_max_add =min_abs + max_abs

    bool_min_max_less_zero = _less_compare_float32(min_max_add, tensor_zero)
    bool_min_max_more_zero = _less_compare_float32(tensor_zero, min_max_add)
    bool_both_no_zero = bool_min_max_less_zero + bool_min_max_more_zero

    return bool_both_no_zero

def _nudged_min_max_compute(min_broadcast, max_broadcast, num_bits,
                            narrow_range):
    if narrow_range is False:
        quant_min = 0
    else:
        quant_min = 1
    quant_max = 2 ** num_bits - 1
    tensor_zero = min_broadcast * 0.0
    quant_min_float = tensor_zero + quant_min
    quant_max_float = tensor_zero + quant_max
    max_sub_min = max_broadcast - min_broadcast
    quant_max_sub_quant_min = quant_max_float - quant_min_float
    scale = max_sub_min / quant_max_sub_quant_min
    min_div_scale = min_broadcast / scale

    zero_point_from_min = quant_min_float - min_div_scale
    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min, quant_min_float)
    bool_more_quant_max_float = _less_compare_float32(quant_max_float, zero_point_from_min)
    less_quant_min_float = quant_min_float * bool_less_quant_min_float
    more_quant_max_float = quant_max_float * bool_more_quant_max_float
    bool_not_less_quant_min_float = _bool_negate(bool_less_quant_min_float)
    bool_not_more_quant_max_float = _bool_negate(bool_more_quant_max_float)

    bool_between_min_max = bool_not_less_quant_min_float * bool_not_more_quant_max_float
    between_min_max_float = zero_point_from_min * bool_between_min_max
    # use DSL floor(x+0.5) to implement the round(x) function of the tf
    between_min_max_add_half_one = between_min_max_float + 0.5
    between_min_max_round = np.floor(between_min_max_add_half_one)
    nudged_zero_point_tensor = less_quant_min_float + more_quant_max_float
    nudged_zero_point = nudged_zero_point_tensor + between_min_max_round

    nudged_min_tensor = quant_min_float - nudged_zero_point
    nudged_min = nudged_min_tensor * scale

    tensor_zero_second = min_broadcast * 0.0
    quant_min_float_second = tensor_zero_second + quant_min
    quant_max_float_second = tensor_zero_second + quant_max
    max_sub_min_second = max_broadcast - min_broadcast
    quant_max_sub_quant_min_second = quant_max_float_second - quant_min_float_second
    scale_second = max_sub_min_second / quant_max_sub_quant_min_second
    min_div_scale_second = min_broadcast / scale_second
    zero_point_from_min_second = quant_min_float_second - min_div_scale_second
    bool_less_quant_min_second = _less_compare_float32(zero_point_from_min_second, quant_min_float_second)
    bool_more_quant_max_second = _less_compare_float32(quant_max_float_second, zero_point_from_min_second)
    less_quant_min_float_second = quant_min_float_second * bool_less_quant_min_second
    more_quant_max_float_second = quant_max_float_second * bool_more_quant_max_second
    bool_not_less_quant_min_second = _bool_negate(bool_less_quant_min_second)
    bool_not_more_quant_max_second = _bool_negate(bool_more_quant_max_second)
    bool_between_min_max_second = bool_not_less_quant_min_second * bool_not_more_quant_max_second
    between_min_max_float_second = zero_point_from_min_second * bool_between_min_max_second
    min_max_add_half_one_second = between_min_max_float_second + 0.5
    between_min_max_round_second = np.floor(min_max_add_half_one_second)
    nudged_zero_point_tensor_second = less_quant_min_float_second + more_quant_max_float_second
    nudged_zero_point_second = nudged_zero_point_tensor_second +between_min_max_round_second
    nudged_max_tensor = quant_max_float_second - nudged_zero_point_second
    nudged_max = nudged_max_tensor * scale_second

    return nudged_min, nudged_max

def _between_nudged_min_max_compute(x, nudged_min, nudged_max):
    min_value = 2.0 ** (-126)
    max_value = 2.0 ** (62)
    factor_value = 2.0 ** (2)

    sub_tensor_min = x - nudged_min
    sub_min = sub_tensor_min + min_value
    more_nudged_min_tensor = np.maximum(sub_min, 0)

    sub_tensor_max = nudged_max - x
    sub_max = sub_tensor_max + min_value
    less_nudged_max_tensor = np.maximum(sub_max, 0)

    between_nudged_tensor = more_nudged_min_tensor * less_nudged_max_tensor
    between_nudged_element = np.minimum(between_nudged_tensor, min_value)

    vmul_max_value = between_nudged_element * max_value
    vmul_factor_value = vmul_max_value * max_value
    between_nudged = vmul_factor_value * factor_value
    return between_nudged

def calc_expect_func(gradients, x, min, max, backprops_wrt_x,
                     backprops_wrt_min, backprops_wrt_max, num_bits, narrow_range):
    shape = list(x['shape'])
    dtype = x['dtype']
    axis = []
    for i, _ in enumerate(shape):
        axis.append(i)
    min_broadcast = np.broadcast_to(min['value'], shape)
    max_broadcast = np.broadcast_to(max['value'], shape)
    nudged_min, nudged_max = _nudged_min_max_compute(min_broadcast,
                                                     max_broadcast, num_bits,
                                                     narrow_range)
    nudged_min_backup = nudged_min + 0.0
    nudged_max_backup = nudged_max + 0.0
    between_nudged_min_max = _between_nudged_min_max_compute(x['value'], nudged_min,
                                                             nudged_max)
    wrt_input_tensor = between_nudged_min_max * gradients['value']

    bool_below_min = _less_compare_float32(x['value'], nudged_min_backup)
    below_min_data = bool_below_min * gradients['value']
    bool_below_max = _less_compare_float32(nudged_max_backup, x['value'])
    below_max_data = bool_below_max * gradients['value']
    bool_both_no_zero = _both_min_max_zero(min['value'], max['value'], shape, dtype)
    bool_both_no_zero_reverse = np.ones(shape) - bool_both_no_zero

    wrt_input_weight = wrt_input_tensor * bool_both_no_zero
    gradients_weight = gradients['value'] * bool_both_no_zero_reverse
    backprops_wrt_x = wrt_input_weight + gradients_weight

    temp_insert_node_mul =  backprops_wrt_x * 0.0 + below_min_data
    below_min_data_tensor = temp_insert_node_mul * bool_both_no_zero
    below_max_data_tensor = below_max_data * bool_both_no_zero
    backprop_wrt_min = np.sum(below_min_data_tensor, tuple(axis))
    backprop_wrt_max = np.sum(below_max_data_tensor, tuple(axis))

    return backprops_wrt_x, backprop_wrt_min, backprop_wrt_max

# ut_case.add_precision_case("all", {"params": [{"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0,0.5]},
#                                               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "input","value_range":[0.5,1.0]},
#                                               {"shape": (4, 2), "dtype": "float32", "format": "ND", "ori_shape": (4, 2),"ori_format": "ND", "param_type": "output"},
#                                               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
#                                               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
#                                               8,True],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })

