#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("INTrainingReduceV2 ", "impl.in_training_reduce_v2", "in_training_reduce_v2")


# TODO add you test case

def verify_in_training_reduce_v2(shape_x, shape_result,
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


ut_case.add_case("all",
                 verify_in_training_reduce_v2((6, 5, 8, 7, 6), (1, 1, 1, 1, 1),
                                              "NC2HWC0",
                                              "float32",
                                              "float32",
                                              "test_error_001",
                                              RuntimeError))

ut_case.add_case("all",
                 verify_in_training_reduce_v2((6, 5, 8, 7), (1, 1, 1, 1),
                                              "NC1HWC0",
                                              "float32",
                                              "float32",
                                              "test_error_002",
                                              RuntimeError))

ut_case.add_case("all",
                 verify_in_training_reduce_v2((6, 5, 8, 7, 16), (6, 5, 1, 1, 16),
                                              "NC1HWC0",
                                              "float32",
                                              "float32",
                                              "test_right_001",
                                              "success"))

ut_case.add_case("all",
                 verify_in_training_reduce_v2((6, 5, 8, 7, 16), (6, 5, 1, 1, 16),
                                              "NC1HWC0",
                                              "float16",
                                              "float32",
                                              "test_right_002",
                                              "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
