#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("MaximumGrad", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "maximum_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "maximum_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "maximum_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "maximum_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "maximum_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

#precision cases
def _compare_value(data_x, data_y):
    min_value = 2 ** (-126)
    max_value = 2 ** (62)
    max_value_1 = 2 ** 2
    data_zero = data_x * 0.0
    min_value_tensor = data_zero + min_value
    max_value_tensor = data_zero + max_value
    max_value_1_tensor = data_zero + max_value_1
    sub_xy = data_x - data_y
    add_min_value = sub_xy + min_value
    vmax_zero = np.maximum(add_min_value, data_zero)
    vmin_min_value = np.minimum(vmax_zero, min_value_tensor)
    vmul_max_value = vmin_min_value * max_value_tensor
    vmul_max_value_1 = vmul_max_value * max_value_tensor
    result = vmul_max_value_1 * max_value_1_tensor
    return result

def _calculate_result_ge(data_x, data_y, data_dz, dtype, shape_dz):
    ones = np.ones(shape_dz)
    minus_one_tensor = -1 * ones
    datax_select_ge = _compare_value(data_x, data_y)
    result_dx = data_dz * datax_select_ge
    result_dy = (datax_select_ge + minus_one_tensor) * minus_one_tensor * data_dz
    return result_dx, result_dy
    
def calc_expect_func(grads, x1, x2, y1, y2):
    dtype = x1['dtype']
    shape_dz = grads['shape']
    result_dx, result_dy = _calculate_result_ge(x1['value'], x2['value'], grads['value'],
                                                dtype, shape_dz)
    return result_dx, result_dy

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "output"},
                                                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "output"},
                                                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (3, 4, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 64),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (3, 4, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 64),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (3, 4, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 64),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (3, 4, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 64),"ori_format": "ND", "param_type": "output"},
                                                    {"shape": (3, 4, 64), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 64),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 2, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 2, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 2, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 2, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 2, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 2, 16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 2, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 2, 16),"ori_format": "ND", "param_type": "output"},
                                                    {"shape": (16, 2, 2, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 2, 16),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (10, 33), "dtype": "float32", "format": "ND", "ori_shape": (10, 33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (10, 33), "dtype": "float32", "format": "ND", "ori_shape": (10, 33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (10, 33), "dtype": "float32", "format": "ND", "ori_shape": (10, 33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (10, 33), "dtype": "float32", "format": "ND", "ori_shape": (10, 33),"ori_format": "ND", "param_type": "output"},
                                                    {"shape": (10, 33), "dtype": "float32", "format": "ND", "ori_shape": (10, 33),"ori_format": "ND", "param_type": "output"}],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
