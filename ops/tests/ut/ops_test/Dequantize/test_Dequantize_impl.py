#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf

ut_case = OpUT("Dequantize", None, None)

def calc_expect_func(x, min, max, y, mode):
    inputx = tf.cast(x['value'], dtype="qint8")
    min_range = tf.reshape(min['value'], [])
    max_range = tf.reshape(max['value'], [])
    output_data = tf.dequantize(inputx, min_range, max_range, mode)
    with tf.compat.v1.Session() as sess:
        output_data = sess.run(output_data)
        return output_data

case1 = {"params": [{"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    "MIN_COMBINED"],
         "case_name": "dequantize_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    "MIN_FIRST"],
         "case_name": "dequantize_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND"},
                    "SCALED"],
         "case_name": "dequantize_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

case4 = {"params": [{"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND","param_type":"input"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","param_type":"input"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND","param_type":"input"},
                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND","param_type":"output"},
                    "SCALED"],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}


ut_case.add_precision_case("Ascend910", case4)

if __name__ == '__main__':
    ut_case.run()
