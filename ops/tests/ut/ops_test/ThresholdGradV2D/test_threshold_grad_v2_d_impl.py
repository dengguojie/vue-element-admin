#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf

ut_case = OpUT("ThresholdGradV2D", "impl.threshold_grad_v2_d", "threshold_grad_v2_d")

def calc_expect_func(gradients, features, output, threshold):

    greater = tf.greater(features["value"], threshold)
    data = tf.where(greater, gradients["value"], tf.constant(0, src_type, gradients["shape"]))
    session = tf.Session()
    output = session.run(data)

    return output

# pylint: disable=locally-disabled,too-many-arguments
def gen_threshold_grad_v2_d_case(shape_x, shape_y, dtype_val1, dtype_val2,
                                 format_var, threshold,
                                 kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "format": format_var,
                        "dtype": dtype_val1, "ori_shape": shape_x,
                        "ori_format": format_var, "param_type":"input"},
                       {"shape": shape_y, "format": format_var,
                        "dtype": dtype_val2, "ori_shape": shape_y,
                        "ori_format": format_var, "param_type":"input"},
                       {"shape": shape_x, "format": format_var,
                        "dtype": dtype_val1, "ori_shape": shape_x,
                        "ori_format": format_var, "param_type":"output"},
                       threshold],
            "case_name": kernel_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}

ut_case.add_precision_case("all",
                 gen_threshold_grad_v2_d_case((5, 1, 2, 2), (5, 1, 2, 2),
                                              "float16", "float16",
                                              "ND", 1.0,
                                              "test_float16_case",
                                              "success"))
ut_case.add_precision_case("all",
                 gen_threshold_grad_v2_d_case((1, 1, 2, 2), (1, 1, 2, 2),
                                              "int8", "int8",
                                              "ND", 1.2,
                                              "test_int8_case",
                                              "success"))
ut_case.add_precision_case("all",
                 gen_threshold_grad_v2_d_case((2, 2), (2, 2),
                                              "int32", "int32",
                                              "ND", 1.3,
                                              "test_int32_case",
                                              "success"))
ut_case.add_precision_case("all",
                 gen_threshold_grad_v2_d_case((1, 1, 2, 2), (1, 1, 2, 2),
                                              "uint8", "uint8",
                                              "ND", 1.0,
                                              "test_uint8_case",
                                              "success"))
ut_case.add_case("all",
                 gen_threshold_grad_v2_d_case((1,), (1,),
                                              "float16", "float16",
                                              "ND", 1.0,
                                              "test_1dim_case",
                                              "success"))
ut_case.add_case("all",
                 gen_threshold_grad_v2_d_case((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2),
                                              "float16", "float16",
                                              "ND", 1.0,
                                              "test_8dim_case",
                                              "success"))
ut_case.add_case("all",
                 gen_threshold_grad_v2_d_case((199999999, 1), (199999999, 1),
                                              "float16", "float16",
                                              "ND", 1.0,
                                              "test_large_shapeNum_case1",
                                              "success"))
ut_case.add_case("all",
                 gen_threshold_grad_v2_d_case((1, 1, 2, 2), (1, 1, 2, 5),
                                              "float16", "float16",
                                              "ND", 1.0,
                                              "test_different_shape_case",
                                              RuntimeError))
ut_case.add_case("all",
                 gen_threshold_grad_v2_d_case((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2, 2),
                                              "float16", "float16",
                                              "ND", 1.0,
                                              "test_9dim_case",
                                              RuntimeError))

if __name__ == '__main__':
    ut_case.run("Ascend910")
