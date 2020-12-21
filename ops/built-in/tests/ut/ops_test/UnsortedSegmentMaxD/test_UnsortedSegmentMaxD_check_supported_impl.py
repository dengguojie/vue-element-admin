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

TransposeD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("UnsortedSegmentMaxD", "impl.unsorted_segment_max_d", "check_supported")

case_num_segments_error = {
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
            0
        ],
    "case_name": 'case_num_segments_error',
    "expect": RuntimeError
}

case_first_shape_error = {
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
                "shape": (31, ),  # segment_ids
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
    "case_name": 'case_first_shape_error',
    "expect": RuntimeError
}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case_num_segments_error)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case_first_shape_error)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
