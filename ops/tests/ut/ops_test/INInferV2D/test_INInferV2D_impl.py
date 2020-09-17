#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("InInferV2d", None, None)


def verify_in_infer_v2(shape_x, shape_weight, shape_mean,
                       data_format, 
                       epsilon, 
                       dtype, dtype_others,
                       kernel_name, expect):
    if shape_weight == (0,):
        return {"params":
                    [{"shape": shape_x, "dtype": dtype, "format": data_format,
                      "ori_shape": shape_x, "ori_format": data_format},
                     None, None,
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_x, "dtype": dtype, "format": data_format,
                      "ori_shape": shape_x, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format}],
                "case_name": kernel_name,
                "expect": expect}
    else:
        return {"params":
                    [{"shape": shape_x, "dtype": dtype, "format": data_format,
                      "ori_shape": shape_x, "ori_format": data_format},
                     {"shape": shape_weight, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_weight, "ori_format": data_format},
                     {"shape": shape_weight, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_weight, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_x, "dtype": dtype, "format": data_format,
                      "ori_shape": shape_x, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format},
                     {"shape": shape_mean, "dtype": dtype_others,
                      "format": data_format,
                      "ori_shape": shape_mean, "ori_format": data_format}],
                "case_name": kernel_name,
                "expect": expect}

ut_case.add_case("all",
                 verify_in_infer_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
                                    (6, 5, 1, 1, 6),
                                    "NC1HWC0", 0.0001,
                                    "float64",
                                    "float32",
                                    "test_error_001", 
                                    RuntimeError))

ut_case.add_case("all",
                 verify_in_infer_v2((6, 5, 8, 7), (6, 5, 1, 1, 6),
                                    (6, 5, 1, 1, 6),
                                    "NC1HWC0", 0.0001,
                                    "float32",
                                    "float32",
                                    "test_error_002", 
                                    RuntimeError))

ut_case.add_case("all",
                 verify_in_infer_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 5),
                                    (6, 5, 1, 1, 6),
                                    "NC1HWC0", 0.0001,
                                    "float32",
                                    "float32",
                                    "test_error_003", 
                                    RuntimeError))

ut_case.add_case("all",
                 verify_in_infer_v2((6, 5, 8, 7, 6), (6, 5, 2, 1, 6),
                                    (6, 5, 1, 1, 6),
                                    "NC1HWC0", 0.0001,
                                    "float32",
                                    "float32",
                                    "test_error_004", 
                                    RuntimeError))

ut_case.add_case("all",
                 verify_in_infer_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
                                    (6, 5, 1, 1, 6),
                                    "NCHW", 0.0001,
                                    "float32",
                                    "float32",
                                    "test_error_005", 
                                    RuntimeError))

ut_case.add_case("all",
                 verify_in_infer_v2((6, 5, 1, 1, 16), (0,),
                                    (6, 5, 1, 1, 16),
                                    "NC1HWC0", 0.0001,
                                    "float32",
                                    "float32", 
                                    "test_right_wo_gamma_001", 
                                    "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)
