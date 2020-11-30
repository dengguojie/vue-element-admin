#!/usr/bin/env python
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

SliceDV2 ut case
"""
import numpy as np
import tensorflow as tf

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("SliceDV2", "impl.slice_d_v2", "slice_d_v2")

case1 = {
    "params": [{"shape": (5, 13, 4), "dtype": "int32", "format": "NCHW", "ori_shape": (5, 13, 4), "ori_format": "NCHW"},
               {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
               {"shape": (2, 12, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 12, 3), "ori_format": "NCHW"},
               (2, -1, -1),
               ],
    "case_name": "SliceDV2_1",
    "expect": "success",
    "support_expect": True}

case2 = {
    "params": [{"shape": (65, 75), "dtype": "float32", "format": "NCHW", "ori_shape": (65, 75), "ori_format": "NCHW"},
               {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
               {"shape": (15, 33), "dtype": "float32", "format": "NCHW", "ori_shape": (15, 33), "ori_format": "NCHW"},
               (15, 33),
               ],
    "case_name": "SliceDV2_2",
    "expect": "success",
    "support_expect": True}

case3 = {"params": [
    {"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (2, 4, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 4, 3, 1), "ori_format": "NCHW"},
    (2, 4, 3, 1),
],
    "case_name": "SliceDV2_3",
    "expect": "success",
    "support_expect": True}

case4 = {"params": [
    {"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (1, 1, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 3, 1), "ori_format": "NCHW"},
    (1, 1, 3, 1),
],
    "case_name": "SliceDV2_4",
    "expect": "success",
    "support_expect": True}

case5 = {"params": [
    {"shape": (13, 7, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 1, 1), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (2, 2, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 1), "ori_format": "NCHW"},
    (2, 2, 1, 1),
],
    "case_name": "SliceDV2_5",
    "expect": "success",
    "support_expect": True}

case6 = {"params": [
    {"shape": (13, 7, 5, 5), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 5), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (1, 1, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 1, 1), "ori_format": "NCHW"},
    (1, 1, 1, 1),
],
    "case_name": "SliceDV2_6",
    "expect": "success",
    "support_expect": True}

case7 = {"params": [
    {"shape": (2, 70000), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 70000), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (2, 69999), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 69999), "ori_format": "NCHW"},
    (2, 69999),
],
    "case_name": "SliceDV2_7",
    "expect": "success",
    "support_expect": True}

case8 = {"params": [
    {"shape": (7, 200, 600), "dtype": "float16", "format": "NCHW", "ori_shape": (7, 200, 600), "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
    {"shape": (3, 128, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 128, 512), "ori_format": "NCHW"},
    (3, 128, 512),
],
    "case_name": "SliceDV2_8",
    "expect": "success",
    "support_expect": True}

case9 = {"params": [{"shape": (9, 11, 270000), "dtype": "float16", "format": "NCHW", "ori_shape": (9, 11, 270000),
                     "ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
                    {"shape": (3, 5, 262144), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 5, 262144),
                     "ori_format": "NCHW"},
                    (3, 5, 262144),
                    ],
         "case_name": "SliceDV2_9",
         "expect": "success",
         "support_expect": True}

case10 = {
    "params": [{"shape": (459999,), "dtype": "float16", "format": "NCHW", "ori_shape": (459999,), "ori_format": "NCHW"},
               {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
               {"shape": (458752,), "dtype": "float16", "format": "NCHW", "ori_shape": (458752,), "ori_format": "NCHW"},
               (458752,),
               ],
    "case_name": "SliceDV2_10",
    "expect": "success",
    "support_expect": True}

case11 = {"params": [
    {"shape": (65536, 31748), "dtype": "int64", "format": "NCHW", "ori_shape": (65536, 31748), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (0, 0), "dtype": "int64", "format": "NCHW", "ori_shape": (0, 0), "ori_format": "NCHW"},
    (65536, 31748),
],
    "case_name": "SliceDV2_11",
    "expect": "success",
    "support_expect": True}

case12 = {"params": [
    {"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"},
    (160000, 16),
],
    "case_name": "SliceDV2_12",
    "expect": "success",
    "support_expect": True}

case13 = {"params": [
    {"shape": (13, 7, 5, 5), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 5), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int8", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (1, 1, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 1, 1), "ori_format": "NCHW"},
    (1, 1, 1, 1),
],
    "case_name": "SliceDV2_13",
    "expect": "RuntimeError",
    "support_expect": False}

case14 = {"params": [
    {"shape": (160000, 16), "dtype": "int32", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int8", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (160000, 16), "dtype": "int32", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"},
    (160000, 16),
],
    "case_name": "SliceDV2_14",
    "expect": "RuntimeError",
    "support_expect": False}

case15 = {
    "params": [{"shape": (459999,), "dtype": "float16", "format": "NCHW", "ori_shape": (459999,), "ori_format": "NCHW"},
               {"shape": (3,), "dtype": "int8", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
               {"shape": (458752,), "dtype": "float16", "format": "NCHW", "ori_shape": (458752,), "ori_format": "NCHW"},
               (458752,),
               ],
    "case_name": "SliceDV2_15",
    "expect": "RuntimeError",
    "support_expect": False}

ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend910"], case12)
ut_case.add_case(["Ascend310", "Ascend910"], case13)
ut_case.add_case(["Ascend310", "Ascend910"], case14)
ut_case.add_case(["Ascend310", "Ascend910"], case15)


if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310"])
    exit(0)
