#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("INInferV2", "impl.in_infer_v2", "in_infer_v2")


# TODO add you test case

def gen_in_infer_v2_case(shape_x, shape_weight, shape_mean,
                         data_format,
                         epsilon,
                         dtype, dtype_others,
                         kernel_name, expect):
    return {"params":
                [{"shape": shape_x, "dtype": dtype, "format": data_format,
                  "ori_shape": shape_x, "ori_format": data_format},
                 # gamma
                 {"shape": shape_weight, "dtype": dtype_others,
                  "format": data_format,
                  "ori_shape": shape_weight, "ori_format": data_format
                  } if shape_weight else None,
                 # beta
                 {"shape": shape_weight, "dtype": dtype_others,
                  "format": data_format,
                  "ori_shape": shape_weight, "ori_format": data_format
                  } if shape_weight else None,
                 # mean
                 {"shape": shape_mean, "dtype": dtype_others,
                  "format": data_format,
                  "ori_shape": shape_mean, "ori_format": data_format},
                 # variance
                 {"shape": shape_mean, "dtype": dtype_others,
                  "format": data_format,
                  "ori_shape": shape_mean, "ori_format": data_format},
                 # y
                 {"shape": shape_x, "dtype": dtype, "format": data_format,
                  "ori_shape": shape_x, "ori_format": data_format},
                 # batch_mean
                 {"shape": shape_mean, "dtype": dtype_others,
                  "format": data_format,
                  "ori_shape": shape_mean, "ori_format": data_format},
                 # batch_variance
                 {"shape": shape_mean, "dtype": dtype_others,
                  "format": data_format,
                  "ori_shape": shape_mean, "ori_format": data_format},
                 epsilon],
            "case_name": kernel_name,
            "expect": expect}


def calc_expect_func(input_arr, input_gamma, input_beta,
                     input_mean, input_var,
                     output_y, output_mean, output_variance,
                     epsilon=0.00001):
    dtype_x = input_arr["dtype"]
    input_x = input_arr["value"]

    if dtype_x == "float16":
        input_x = input_x.astype(np.float32)

    result = (input_x - input_mean["value"]) / (
              np.sqrt(input_var["value"] + epsilon))
    if input_gamma is not None and input_beta is not None:
        result = result * input_gamma["value"] + input_beta["value"]
    if dtype_x == "float16":
        result = result.astype(np.float16)

    return result, input_mean["value"], input_var["value"]


def gen_in_infer_v2_precision_case(shape_x, shape_weight, shape_mean,
                         data_format,
                         epsilon,
                         dtype, dtype_others,
                         kernel_name, expect):
    return {
        "params":
            [{"shape": shape_x, "dtype": dtype, "format": data_format,
              "ori_shape": shape_x, "ori_format": data_format,
              "param_type": "input", "value_range": [-10.0, 10.0]},
             # gamma
             {"shape": shape_weight, "dtype": dtype_others,
              "format": data_format,
              "ori_shape": shape_weight, "ori_format": data_format,
              "param_type": "input", "value_range": [-1.0, 1.0]
              } if shape_weight else None,
             # beta
             {"shape": shape_weight, "dtype": dtype_others,
              "format": data_format,
              "ori_shape": shape_weight, "ori_format": data_format,
              "param_type": "input", "value_range": [-1.0, 1.0]
              } if shape_weight else None,
             # mean
             {"shape": shape_mean, "dtype": dtype_others,
              "format": data_format,
              "ori_shape": shape_mean, "ori_format": data_format,
              "param_type": "input", "value_range": [1.0, 10.0]},
             # variance
             {"shape": shape_mean, "dtype": dtype_others,
              "format": data_format,
              "ori_shape": shape_mean, "ori_format": data_format,
              "param_type": "input", "value_range": [1.0, 10.0]},
             # y
             {"shape": shape_x, "dtype": dtype, "format": data_format,
              "ori_shape": shape_x, "ori_format": data_format,
               "param_type": "output"},
             # batch_mean
             {"shape": shape_mean, "dtype": dtype_others,
              "format": data_format,
              "ori_shape": shape_mean, "ori_format": data_format,
               "param_type": "output"},
             # batch_variance
             {"shape": shape_mean, "dtype": dtype_others,
              "format": data_format,
              "ori_shape": shape_mean, "ori_format": data_format,
               "param_type": "output"},
             epsilon],
        "case_name": kernel_name,
        "expect": expect,
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 8, 7), (6, 5, 1, 1, 6),
                                      (6, 5, 1, 1, 6),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_error_002",
                                      RuntimeError))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 8, 7, 6), (6, 5, 2, 1, 6),
                                      (6, 5, 1, 1, 6),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_error_004",
                                      RuntimeError))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
                                      (6, 5, 1, 1, 6),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_error_001",
                                      RuntimeError))
ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 8, 7, 6), (6, 5, 1, 1, 5),
                                      (6, 5, 1, 1, 6),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_error_003",
                                      RuntimeError))
ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
                                      (6, 5, 1, 1, 6),
                                      "NCHW", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_error_005",
                                      RuntimeError))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 8, 7, 16), (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_right_fp32_001",
                                      "success"))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 8, 7, 16), (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float16",
                                      "float32",
                                      "test_right_fp16_002",
                                      "success"))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 18, 7, 16), (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_right_fp32_003",
                                      "success"))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 18, 7, 16), (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float16",
                                      "float32",
                                      "test_right_fp16_004",
                                      "success"))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 18, 7, 16), None,
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float16",
                                      "float32",
                                      "test_right_fp16_005",
                                      "success"))

ut_case.add_case("all",
                 gen_in_infer_v2_case((6, 5, 18, 7, 16), None,
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "test_right_fp16_006",
                                      "success"))

ut_case.add_precision_case("all",
                 gen_in_infer_v2_precision_case((6, 5, 8, 7, 16), 
                                      (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "in_infer_v2_precision_case_001",
                                      "success"))

ut_case.add_precision_case("all",
                 gen_in_infer_v2_precision_case((6, 5, 8, 7, 16), 
                                      (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float16",
                                      "float32",
                                      "in_infer_v2_precision_case_002",
                                      "success"))

ut_case.add_precision_case("all",
                 gen_in_infer_v2_precision_case((6, 5, 18, 7, 16), 
                                      (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float32",
                                      "float32",
                                      "in_infer_v2_precision_case_003",
                                      "success"))

ut_case.add_precision_case("all",
                 gen_in_infer_v2_precision_case((6, 5, 18, 7, 16), 
                                      (6, 5, 1, 1, 16),
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float16",
                                      "float32",
                                      "in_infer_v2_precision_case_004",
                                      "success"))

ut_case.add_precision_case("all",
                 gen_in_infer_v2_precision_case((6, 5, 18, 7, 16), None,
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.0001,
                                      "float16",
                                      "float32",
                                      "in_infer_v2_precision_case_005",
                                      "success"))

ut_case.add_precision_case("all",
                 gen_in_infer_v2_precision_case((6, 5, 18, 7, 16), None,
                                      (6, 5, 1, 1, 16),
                                      "NC1HWC0", 0.00001,
                                      "float32", "float32",
                                      "in_infer_v2_precision_case_006",
                                      "success"))

if __name__ == '__main__':
    ut_case.run("Ascend310")
    ut_case.run(["Ascend310"], simulator_mode="pv", 
                simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
