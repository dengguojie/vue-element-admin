#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("GNTrainingReduce", "impl.gn_training_reduce",
               "gn_training_reduce")


# TODO add you test case

def gen_gn_training_reduce_case(shape_x, shape_sum,
                                num_group, data_format,
                                dtype, dtype_others,
                                kernel_name, expect):
    return {"params":
                [{"shape": shape_x, "ori_shape": shape_x, "dtype": dtype,
                  "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format},
                 num_group],
            "case_name": kernel_name,
            "expect": expect}

def calc_expect_func(inputArr, out_sum, square_sum, num_groups=2):

    format_x = inputArr["format"]
    shape = inputArr["shape"]
    dtype_x = inputArr["dtype"]
    input_x = inputArr["value"]

    if format_x == "NCHW":
        final_shape = (shape[0], num_groups, shape[1] // num_groups, shape[2], shape[3])
        axis = [2, 3, 4]
    else:
        final_shape = (shape[0], shape[1], shape[2], num_groups, shape[3] // num_groups)
        axis = [1, 2, 4]
    axis = tuple(axis)

    if dtype_x == "float16":
        input_x = input_x.astype(np.float32)
    input_x = np.reshape(input_x, final_shape)

    result_sum = np.sum(input_x, axis=axis, keepdims=True)
    square_sum = np.sum(input_x * input_x, axis=axis, keepdims=True)
    print('---sum{}'.format(result_sum))
    return result_sum, square_sum


def gen_gn_training_reduce_precision_case(shape_x, shape_sum,
                                num_group, data_format,
                                dtype, dtype_others,
                                kernel_name, expect):
    return {"params":
                [{"shape": shape_x, "ori_shape": shape_x, "dtype": dtype,
                  "format": data_format, "ori_format": data_format,
                 "param_type": "input", "value_range": [0.0, 1.0]},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format,"param_type": "output"},
                 {"shape": shape_sum, "ori_shape": shape_sum,
                  "dtype": dtype_others, "format": data_format,
                  "ori_format": data_format,"param_type": "output"},
                 num_group],
            "case_name": kernel_name,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 3, 224, 224), (64, 3, 1, 1, 1),
                                             1, "NC2HWC0",
                                             "float32", "float32",
                                             "cce_gn_training_reduce1",
                                             RuntimeError))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 3, 224, 224), (64, 3, 1, 1, 1),
                                             1, "NCHW",
                                             "float64", "float32",
                                             "cce_gn_training_reduce2",
                                             RuntimeError))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 3, 224, 224), (64, 3, 1, 1, 11),
                                             2, "NCHW",
                                             "float32", "float32",
                                             "cce_gn_training_reduce3",
                                             RuntimeError))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 3, 224, 224, 1),
                                             (64, 3, 1, 1, 11),
                                             2, "NCHW",
                                             "float32", "float32",
                                             "cce_gn_training_reduce4",
                                             RuntimeError))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 4, 224, 224), (64, 2, 1, 1, 1),
                                             2, "NCHW",
                                             "float32", "float32",
                                             "cce_gn_training_reduce5",
                                             "success"))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 224, 224, 4), (64, 1, 1, 2, 1),
                                             2, "NHWC",
                                             "float32", "float32",
                                             "cce_gn_training_reduce6",
                                             "success"))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 224, 224, 4), (64, 1, 1, 2, 1),
                                             2, "NHWC",
                                             "float16", "float32",
                                             "cce_gn_training_reduce7",
                                             "success"))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 224, 224, 122),
                                             (64, 1, 1, 2, 1),
                                             2, "NHWC",
                                             "float32", "float32",
                                             "cce_gn_training_reduce8",
                                             "success"))

ut_case.add_case("all",
                 gen_gn_training_reduce_case((64, 224, 224, 120),
                                             (64, 1, 1, 2, 1),
                                             2, "NHWC",
                                             "float16", "float32",
                                             "cce_gn_training_reduce9",
                                             "success"))

# TODO run error
# ut_case.add_precision_case("all",
#                  gen_gn_training_reduce_precision_case((64, 4, 224, 224), (64, 2, 1, 1, 1),
#                                              2, "NCHW",
#                                              "float32", "float32",
#                                              "gn_training_reduce_precision_case_001",
#                                              "success"))
# TODO run error
# ut_case.add_precision_case("all",
#                  gen_gn_training_reduce_precision_case((64, 224, 224, 4), (64, 1, 1, 2, 1),
#                                              2, "NHWC",
#                                              "float32", "float32",
#                                              "gn_training_reduce_precision_case_002",
#                                              "success"))
#
# ut_case.add_precision_case("all",
#                  gen_gn_training_reduce_precision_case((64, 224, 224, 4), (64, 1, 1, 2, 1),
#                                              2, "NHWC",
#                                              "float16", "float32",
#                                              "gn_training_reduce_precision_case_003",
#                                              "success"))
#
# ut_case.add_precision_case("all",
#                  gen_gn_training_reduce_precision_case((64, 224, 224, 122),
#                                              (64, 1, 1, 2, 1),
#                                              2, "NHWC",
#                                              "float32", "float32",
#                                              "gn_training_reduce_precision_case_004",
#                                              "success"))


# TODO run you test case
if __name__ == '__main__':
    ut_case.run("Ascend910")
    ut_case.run(["Ascend910"], simulator_mode="pv",
            simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
