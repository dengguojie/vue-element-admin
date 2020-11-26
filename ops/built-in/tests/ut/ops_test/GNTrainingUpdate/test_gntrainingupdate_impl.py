#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

unit_case = OpUT("GNTrainingUpdate", "impl.gn_training_update",
               "gn_training_update")


# TODO add you test case

def gen_gn_training_update_case(shape_x, shape_sum, shape_square_sum,
                                shape_scale, shape_mean, shape_y,
                                data_format,
                                epsilon, num_groups,
                                dtype, dtype_others,
                                kernel_name, expect):
    return {"params":
                [{"shape": shape_x, "ori_shape": shape_x,
                  "dtype": dtype, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_square_sum,
                  "ori_shape": shape_square_sum, "dtype": dtype_others,
                  "format": data_format, "ori_format": data_format},
                 {"shape": shape_scale, "ori_shape": shape_scale,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_scale, "ori_shape": shape_scale,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_mean, "ori_shape": shape_mean,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_y, "ori_shape": shape_y,
                  "dtype": dtype, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 epsilon, num_groups],
            "case_name": kernel_name,
            "expect": expect}


def calc_expect_func(input_arr, input_sum, square_sum,
                     input_gamma, input_beta,
                     mean, variance,
                     output_y, batch_mean, batch_variance,
                     epsilon=0.0001, num_group=2):
    format_x = input_arr["format"]
    shape = input_arr["shape"]
    dtype_x = input_arr["dtype"]
    input_x = input_arr["value"]

    if format_x == "NCHW":
        final_shape = (shape[0], num_group, 
          shape[1] // num_group, shape[2], shape[3])
        axis = [2, 3, 4]
    else:
        final_shape = (shape[0], shape[1], shape[2], 
          num_group, shape[3] // num_group)
        axis = [1, 2, 4]
    axis = tuple(axis)

    if dtype_x == "float16":
        input_x = input_x.astype(np.float32)
    input_x = np.reshape(input_x, final_shape)

    num = 1
    shape = input_x.shape
    for i in axis:
        num *= shape[i]
    result_mean = input_sum["value"] / num
    result_var = square_sum["value"] / num - result_mean * result_mean
    result = ((input_x - result_mean) / (np.sqrt(result_var + epsilon)))
    result = result * input_gamma["value"] + input_beta["value"]
    if dtype_x == "float16":
        result = result.astype(np.float16)
    print('===mean:{}'.format(result_mean.flatten()[:10]))
    return result, result_mean, result_var


def generate_precision_case(
        shape_x, shape_sum, shape_square_sum,
        shape_scale, shape_mean, shape_y,
        data_format,
        epsilon, num_groups,
        dtype, dtype_others,
        kernel_name, expect):
    return {
        "params":
            [{"shape": shape_x, "ori_shape": shape_x,
              "dtype": dtype, "format": data_format,
              "ori_format": data_format,
              "param_type": "input", "value_range": [-0.0, 0.0]},
             {"shape": shape_sum, "ori_shape": shape_sum,
              "dtype": dtype_others, "format": data_format,
              "ori_format": data_format,
              "param_type": "input", "value_range": [-0.0, 0.0]},
             {"shape": shape_square_sum,
              "ori_shape": shape_square_sum, "dtype": dtype_others,
              "format": data_format, "ori_format": data_format,
              "param_type": "input", "value_range": [1, 10.0]},
             {"shape": shape_scale, "ori_shape": shape_scale,
              "dtype": dtype_others, "format": data_format,
              "ori_format": data_format,
              "param_type": "input", "value_range": [-1.0, 1.0]},
             {"shape": shape_scale, "ori_shape": shape_scale,
              "dtype": dtype_others, "format": data_format,
              "ori_format": data_format,
              "param_type": "input", "value_range": [-10.0, 10.0]},
             {"shape": shape_sum, "ori_shape": shape_sum,
              "dtype": dtype_others, "format": data_format,
              "ori_format": data_format,
              "param_type": "input", "value_range": [-1.0, 1.0]},
             {"shape": shape_sum, "ori_shape": shape_sum,
              "dtype": dtype_others, "format": data_format,
              "ori_format": data_format,
              "param_type": "input", "value_range": [-1.0, 1.0]},
             {"shape": shape_y, "ori_shape": shape_y,
              "dtype": dtype, "format": data_format,
              "ori_format": data_format, "param_type": "output"},
             {"shape": shape_sum, "ori_shape": shape_sum,
              "dtype": dtype_others, "format": data_format,
              "ori_format": data_format, "param_type": "output"},
             {"shape": shape_sum, "ori_shape": shape_sum,
              "dtype": dtype_others, "format": data_format,
              "ori_format": data_format, "param_type": "output"},
             epsilon, num_groups],
        "case_name": kernel_name,
        "expect": expect,
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

unit_case.add_case("all",
                 gen_gn_training_update_case(
                    (6, 5, 8, 7, 6), (6, 5, 1, 1, 3),
                    (6, 5, 1, 1, 3), (6, 5, 1, 1, 3),
                    (6, 5, 1, 1, 3), (6, 5, 1, 1, 3),
                    "NC1HWC0", 0.0001, 2,
                    "float32",
                    "float32",
                    "cce_group_norm_date_formate_error",
                    RuntimeError))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 5, 8, 7), (6, 2, 3, 1, 1),
                                             (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                             (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                             "NCHW", 0.0001, 2,
                                             "float32",
                                             "float32",
                                             "cce_group_norm_channel_error",
                                             RuntimeError))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 1, 1), (6, 2, 3, 1, 1),
                                             (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                             (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                             "NCHW", 0.0001, 2,
                                             "float32",
                                             "float32",
                                             "cce_group_norm_dim_error",
                                             RuntimeError))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 6, 8, 7), (6, 2, 3, 1),
                                             (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                             (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                             "NCHW", 0.0001, 2,
                                             "float32",
                                             "float32",
                                             "cce_group_norm_dim_diff_error",
                                             RuntimeError))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 6, 1, 1), (6, 2, 3, 4, 1),
                                             (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                             (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                             "NCHW", 0.0001, 2,
                                             "float32",
                                             "float32",
                                             "cce_group_norm_dim_diff_error",
                                             RuntimeError))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 8, 7, 6), (6, 1, 43, 2, 3),
                                             (6, 1, 1, 2, 3), (1, 1, 1, 2, 3),
                                             (6, 1, 1, 2, 3), (6, 1, 1, 2, 3),
                                             "NHWC", 0.0001, 2,
                                             "float32",
                                             "float32",
                                             "cce_group_norm_dim_diff_error",
                                             RuntimeError))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 8, 7, 6), (6, 1, 1, 2, 3),
                                             (6, 1, 3, 2, 3), (1, 1, 1, 2, 3),
                                             (6, 1, 1, 2, 3), (6, 1, 1, 2, 3),
                                             "NHWC", 0.0001, 2,
                                             "float32",
                                             "float32",
                                             "cce_group_norm_dim_diff_error",
                                             RuntimeError))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 6, 8, 7), (6, 2, 1, 1, 1),
                                             (6, 2, 1, 1, 1), (1, 2, 1, 1, 1),
                                             (6, 2, 1, 1, 1), (6, 2, 3, 8, 7),
                                             "NCHW", 0.0001, 2,
                                             "float32",
                                             "float32",
                                             "cce_group_norm_dtype_error",
                                             "success"))

unit_case.add_case("all",
                 gen_gn_training_update_case((6, 6, 8, 7), (6, 2, 1, 1, 1),
                                             (6, 2, 1, 1, 1), (1, 2, 1, 1, 1),
                                             (6, 2, 1, 1, 1), (6, 2, 3, 8, 7),
                                             "NCHW", 0.0001, 2,
                                             "float16",
                                             "float32",
                                             "cce_group_norm_dtype_error",
                                             "success"))

unit_case.add_case("all",
                 gen_gn_training_update_case(
                   (8, 11, 3, 24), (8, 1, 1, 2, 1),
                   (8, 1, 1, 2, 1), (1, 1, 1, 2, 1),
                   (8, 1, 1, 2, 1), (8, 11, 3, 2, 12),
                   "NHWC", 0.0001, 2,
                   "float32",
                   "float32",
                   "cce_group_norm_dtype_error",
                   "success"))
unit_case.add_case("all",
                 gen_gn_training_update_case(
                   (8, 11, 3, 24), (8, 1, 1, 2, 1),
                   (8, 1, 1, 2, 1), (1, 1, 1, 2, 1),
                   (8, 1, 1, 2, 1), (8, 11, 3, 2, 12),
                   "NHWC", 0.0001, 2,
                   "float16",
                   "float32",
                   "cce_group_norm_dtype_error",
                   "success"))

unit_case.add_precision_case("all",
                           generate_precision_case(
                             (16, 32, 11, 18), (16, 1, 1, 2, 1),
                             (16, 1, 1, 2, 1), (1, 1, 1, 2, 1),
                             (16, 1, 1, 2, 1), (16, 32, 11, 2, 9),
                             "NHWC", 0.0001, 2,
                             "float32", "float32",
                             "gen_gn_training_update_precision_case_001",
                             "success"))


if __name__ == '__main__':
    unit_case.run("Ascend910")
    unit_case.run(["Ascend910"], simulator_mode="pv",
            simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
