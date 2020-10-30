# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

UnsortedSegmentMaxD ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf

ut_case = OpUT("UnsortedSegmentMaxD")

def calc_expect_func(x, segment_ids, y, num_segments):
    output = tf.math.unsorted_segment_max(x['value'],
                                          segment_ids['value'],
                                          num_segments,
                                          name="output")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_data = sess.run(output)
        return output_data

case_small_shape_scalar_fp32 = {
    "params":
        [
            {
                "shape": (16, ),  # data
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, ),
                "ori_format": "ND"
            },
            {
                "shape": (16, ),  # segment_ids
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16, ),
                "ori_format": "ND"
            },
            {
                "shape": (5, ),  # y
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (5, ),
                "ori_format": "ND"
            },
            # num_segments
            5
        ],
    "case_name": 'test_unsorted_segment_max_d_small_shape_scalar_fp32',
    "expect": "success"
}

case_small_shape_fp32 = {
    "params":
        [
            {
                "shape": (32, 33),  # data
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 33),
                "ori_format": "ND"
            },
            {
                "shape": (32, ),  # segment_ids
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (32, ),
                "ori_format": "ND"
            },
            {
                "shape": (5, 33),  # y
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (5, 33),
                "ori_format": "ND"
            },
            # num_segments
            5
        ],
    "case_name": 'test_unsorted_segment_max_d_small_shape_fp32',
    "expect": "success"
}
precision_case1 = {
    "params":
        [
            {
                "shape": (32, 33),  # data
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (32, 33),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (32, ),  # segment_ids
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (32, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (5, 33),  # y
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (5, 33),
                "ori_format": "ND",
                "param_type": "output"
            },
            # num_segments
            5
        ],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
}

precision_case2 = {
    "params":
        [
            {
                "shape": (16, ),  # data
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (16, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (16, ),  # segment_ids
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16, ),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (5, ),  # y
                "format": "NC1HWC0",
                "dtype": "float32",
                "ori_shape": (5, ),
                "ori_format": "ND",
                "param_type": "output"
            },
            # num_segments
            5
        ],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_scalar_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_fp32)

ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
