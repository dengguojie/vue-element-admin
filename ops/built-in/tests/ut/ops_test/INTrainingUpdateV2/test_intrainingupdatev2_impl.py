#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("INTrainingUpdateV2", "impl.in_training_update_v2",
               "in_training_update_v2")


# TODO add you test case

def gen_in_training_update_v2_case(shape_x, shape_sum, shape_sqrt_sum,
                                   shape_weight, shape_mean,
                                   data_format,
                                   momentum, epsilon, opt_weight, opt_mean,
                                   dtype, dtype_others,
                                   kernel_name, expect):
    return {"params":
                [{"shape": shape_x, "ori_shape": shape_x,
                  "dtype": dtype, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_sqrt_sum, "ori_shape": shape_sqrt_sum,
                  "dtype": dtype_others,
                  "format": data_format, "ori_format": data_format},
                 {"shape": shape_weight, "ori_shape": shape_weight,
                  "dtype": dtype_others,
                  "format": data_format, "ori_format": data_format} if opt_weight else None,
                 {"shape": shape_weight, "ori_shape": shape_weight,
                  "dtype": dtype_others,
                  "format": data_format, "ori_format": data_format} if opt_weight else None,
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format} if opt_mean else None,
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format} if opt_mean else None,
                 {"shape": shape_x, "ori_shape": shape_x,
                  "dtype": dtype, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 momentum, epsilon],
            "case_name": kernel_name,
            "expect": expect}

def calc_expect_func(inputArr, input_sum, input_square_sum, 
                     input_gamma=None, input_beta=None, 
                     input_mean=None, input_var=None,
                     y=None, batch_mean=None, batch_variance=None,
                     momentum=0.1, epsilon=0.00001):
    format_x = inputArr["format"]
    shape = inputArr["shape"]
    dtype_x = inputArr["dtype"]
    input_x = inputArr["value"]

    if format_x == "NC1HWC0":
        axis = [2, 3]
    else:
        H = format_x.index("H")
        W = format_x.index("W")
        axis = [H, W]
    axis = tuple(axis)

    if dtype_x == "float16":
        input_x = input_x.astype(np.float32)

    num = 1
    for i in axis:
        num *= shape[i]
    current_mean = input_sum["value"] / num
    current_var = input_square_sum["value"] / num - current_mean * current_mean

    result = ((input_x - current_mean) / (np.sqrt(current_var + epsilon)))
    if input_gamma is not None and input_beta is not None:
        result = result * input_gamma["value"] + input_beta["value"]
    result_mean = current_mean

    if num == 1:
        batch_var_scalar = 0.0
    else:
        batch_var_scalar = float(num) / (num - 1)
    result_var = current_var * batch_var_scalar
    if input_mean is not None and input_var is not None:
        factor_reverse = 1.0 - momentum
        mean_mul = result_mean * momentum
        mean_mul_rev = input_mean["value"] * factor_reverse
        result_mean = mean_mul + mean_mul_rev

        var_mul = result_var * momentum
        mean_var_rev = input_var["value"] * factor_reverse
        result_var = var_mul + mean_var_rev

    if dtype_x == "float16":
        result = result.astype(np.float16)

    return result, result_mean, result_var

def gen_in_training_update_v2_precision_case(shape_x, shape_sum, shape_sqrt_sum,
                                   shape_weight, shape_mean,
                                   data_format,
                                   momentum, epsilon, opt_weight, opt_mean,
                                   dtype, dtype_others,
                                   kernel_name, expect):
    return {"params":
                [{"shape": shape_x, "ori_shape": shape_x,
                  "dtype": dtype, "format": data_format,
                  "ori_format": data_format,
                  "param_type": "input", "value_range": [-10.0, 10.0]},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format,
                  "param_type": "input", "value_range": [-1.0, 1.0]},
                 {"shape": shape_sqrt_sum, "ori_shape": shape_sqrt_sum,
                  "dtype": dtype_others,
                  "format": data_format, "ori_format": data_format,
                  "param_type": "input", "value_range": [1.0, 10.0]},
                 {"shape": shape_weight, "ori_shape": shape_weight,
                  "dtype": dtype_others,
                  "format": data_format, "ori_format": data_format,
                  "param_type": "input", "value_range": [-1.0, 1.0]} if opt_weight else None,
                 {"shape": shape_weight, "ori_shape": shape_weight,
                  "dtype": dtype_others,
                  "format": data_format, "ori_format": data_format,
                  "param_type": "input", "value_range": [-1.0, 1.0]} if opt_weight else None,
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format,
                  "param_type": "input", "value_range": [-1.0, 1.0]} if opt_mean else None,
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format,
                  "param_type": "input", "value_range": [1.0, 10.0]} if opt_mean else None,
                 {"shape": shape_x, "ori_shape": shape_x,
                  "dtype": dtype, "format": data_format,
                  "ori_format": data_format, "param_type": "output"},
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format, "param_type": "output"},
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format, "param_type": "output"},
                 momentum, epsilon],
            "case_name": kernel_name,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.008, 0.008)}

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 8, 7, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, True, True,
                                                "float64",
                                                "float32",
                                                "test_error_001",
                                                RuntimeError))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 8, 7), (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, True, True,
                                                "float32",
                                                "float32",
                                                "test_error_002",
                                                RuntimeError))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 8, 7, 6),
                                                (6, 5, 1, 1, 5),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, True, True,
                                                "float32",
                                                "float32",
                                                "test_error_003",
                                                RuntimeError))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 8, 7, 6),
                                                (6, 5, 2, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, True, True,
                                                "float32",
                                                "float32",
                                                "test_error_004",
                                                RuntimeError))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 8, 7, 6),
                                                (6, 5, 2, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC2HWC0", 0.001, 0.0001, True, True,
                                                "float32",
                                                "float32",
                                                "test_error_005",
                                                RuntimeError))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 8, 7, 6),
                                                (6, 5, 2, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, True, True,
                                                "float64",
                                                "float32",
                                                "test_error_006",
                                                RuntimeError))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 8, 7, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6), ((6, 5, 1, 1, 6)),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, True, True,
                                                "float32",
                                                "float32",
                                                "test_right_007",
                                                "success"))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, True, True,
                                                "float16",
                                                "float32",
                                                "test_right_008",
                                                "success"))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, False, True,
                                                "float16",
                                                "float32",
                                                "test_right_009",
                                                "success"))

ut_case.add_case("all",
                 gen_in_training_update_v2_case((6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, False, False,
                                                "float16",
                                                "float32",
                                                "test_right_010",
                                                "success"))

ut_case.add_precision_case(["Ascend910"],
                 gen_in_training_update_v2_precision_case((6, 15, 1, 1, 6),
                                                (6, 15, 1, 1, 6),
                                                (6, 15, 1, 1, 6), ((6, 15, 1, 1, 6)),
                                                (6, 15, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, False, False,
                                                "float32",
                                                "float32",
                                                "in_training_update_v2_precision_case_001",
                                                "success"))

ut_case.add_precision_case(["Ascend910"],
                 gen_in_training_update_v2_precision_case((6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.00001, True, True,
                                                "float16",
                                                "float32",
                                                "in_training_update_v2_precision_case_002",
                                                "success"))
ut_case.add_precision_case(["Ascend910"],
                 gen_in_training_update_v2_precision_case((6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.0001, False, True,
                                                "float16",
                                                "float32",
                                                "in_training_update_v2_precision_case_003",
                                                "success"))

ut_case.add_precision_case(["Ascend910"],
                 gen_in_training_update_v2_precision_case((6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
                                                (6, 5, 1, 1, 6),
                                                "NC1HWC0", 0.001, 0.00001, False, False,
                                                "float16",
                                                "float32",
                                                "in_training_update_v2_precision_case_004",
                                                "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
