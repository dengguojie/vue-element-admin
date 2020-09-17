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

UnsortedSegmentMinD ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("UnsortedSegmentMinD")

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
    "case_name": 'test_unsorted_segment_min_d_small_shape_scalar_fp32',
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
    "case_name": 'test_unsorted_segment_min_d_small_shape_fp32',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_scalar_fp32)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_fp32)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
