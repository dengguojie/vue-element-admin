#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

unit_case = OpUT("INTrainingReduceV2", "impl.in_training_reduce_v2",
               "in_training_reduce_v2")


# TODO add you test case

def gen_in_training_reduce_v2_case(shape_x, shape_result,
                                   data_format,
                                   dtype, dtype_others,
                                   kernel_name, expect):
    return {"params":
        [
            {
                "shape": shape_x,
                "ori_shape": [shape_x[0], shape_x[1] * shape_x[4], shape_x[2], shape_x[3]] if len(
                    shape_x) == 5 else shape_x,
                "dtype": dtype,
                "format": data_format,
                "ori_format": "NCHW"
            },
            {
                "shape": shape_result,
                "dtype": dtype_others,
                "ori_shape": [shape_result[0], shape_result[1] * shape_result[4], shape_result[2],
                              shape_result[3]] if len(shape_result) == 5 else shape_result,
                "format": data_format,
                "ori_format": "NCHW"
            },
            {
                "shape": shape_result,
                "ori_shape": [shape_result[0], shape_result[1] * shape_result[4], shape_result[2],
                              shape_result[3]] if len(shape_result) == 5 else shape_result,
                "dtype": dtype_others,
                "format": data_format,
                "ori_format": "NCHW"
            }
        ],
        "case_name": kernel_name,
        "expect": expect}


def calc_expect_func(input_arr, output_sum, square_sum):
    format_x = input_arr["format"]
    shape = input_arr["shape"]
    dtype_x = input_arr["dtype"]
    input_x = input_arr["value"]

    if format_x == "NC1HWC0":
        axis = [2, 3]
    else:
        idx_h = format_x.index("H")
        idx_w = format_x.index("W")
        axis = [idx_h, idx_w]
    axis = tuple(axis)

    if dtype_x == "float16":
        input_x = input_x.astype(np.float32)

    result_sum = np.sum(input_x, axis=axis, keepdims=True)
    square_sum = np.sum(input_x * input_x, axis=axis, keepdims=True)

    return result_sum, square_sum


def generate_precision_case(shape_x, shape_result,
                                   data_format,
                                   dtype, dtype_others,
                                   kernel_name, expect):
    ori_shape = [shape_x[0], shape_x[1] * shape_x[4], shape_x[2], shape_x[3]]
    return {"params":
        [
            {
                "shape": shape_x,
                "ori_shape":  ori_shape if len(
                    shape_x) == 5 else shape_x,
                "dtype": dtype,
                "format": data_format,
                "ori_format": "NCHW",
                "param_type": "input", "value_range": [-1.0, 1.0]
            },
            {
                "shape": shape_result,
                "dtype": dtype_others,
                "ori_shape": ori_shape if len(
                    shape_result) == 5 else shape_result,
                "format": data_format,
                "ori_format": "NCHW",
                "param_type": "output"
            },
            {
                "shape": shape_result,
                "ori_shape": ori_shape if len(
                    shape_result) == 5 else shape_result,
                "dtype": dtype_others,
                "format": data_format,
                "ori_format": "NCHW",
                "param_type": "output"
            }
        ],
        "case_name": kernel_name,
        "expect": expect,
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

unit_case.add_case("all",
                 gen_in_training_reduce_v2_case((6, 5, 8, 7, 6), 
                                                (1, 1, 1, 1, 1),
                                                "NC2HWC0",
                                                "float32",
                                                "float32",
                                                "test_error_001",
                                                RuntimeError))

unit_case.add_case("all",
                 gen_in_training_reduce_v2_case((6, 5, 8, 7), 
                                                (1, 1, 1, 1),
                                                "NC1HWC0",
                                                "float32",
                                                "float32",
                                                "test_error_002",
                                                RuntimeError))

unit_case.add_case("all",
                 gen_in_training_reduce_v2_case((6, 5, 8, 7, 16),
                                                (6, 5, 1, 1, 16),
                                                "NC1HWC0",
                                                "float32",
                                                "float32",
                                                "test_right_001",
                                                "success"))

unit_case.add_case("all",
                 gen_in_training_reduce_v2_case((6, 5, 8, 7, 16),
                                                (6, 5, 1, 1, 16),
                                                "NC1HWC0",
                                                "float16",
                                                "float32",
                                                "test_right_002",
                                                "success"))

unit_case.add_case("all",
                 gen_in_training_reduce_v2_case((6, 5, 8, 123, 16),
                                                (6, 5, 1, 1, 16),
                                                "NC1HWC0",
                                                "float16",
                                                "float32",
                                                "test_right_003",
                                                "success"))

unit_case.add_case("all",
                 gen_in_training_reduce_v2_case((6, 5, 8, 123, 16),
                                                (6, 5, 1, 1, 16),
                                                "NC1HWC0",
                                                "float32",
                                                "float32",
                                                "test_right_004",
                                                "success"))

unit_case.add_precision_case("all",
                 generate_precision_case(
                    (6, 5, 8, 7, 16),
                    (6, 5, 1, 1, 16),
                    "NC1HWC0",
                    "float32",
                    "float32",
                    "in_training_reduce_v2_precision_001",
                    "success"))

unit_case.add_precision_case("all",
                 generate_precision_case(
                    (6, 5, 8, 7, 16),
                    (6, 5, 1, 1, 16),
                    "NC1HWC0",
                    "float16",
                    "float32",
                    "in_training_reduce_v2_precision_002",
                    "success"))

unit_case.add_precision_case("all",
                 generate_precision_case(
                    (6, 5, 8, 123, 16),
                    (6, 5, 1, 1, 16),
                    "NC1HWC0",
                    "float16",
                    "float32",
                    "in_training_reduce_v2_precision_003",
                    "success"))

unit_case.add_precision_case("all",
                 generate_precision_case(
                    (6, 5, 8, 123, 16),
                    (6, 5, 1, 1, 16),
                    "NC1HWC0",
                    "float32",
                    "float32",
                    "in_training_reduce_v2_precision_004",
                    "success"))

if __name__ == '__main__':
    unit_case.run("Ascend310")
    unit_case.run(["Ascend310"], simulator_mode="pv", 
        simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
