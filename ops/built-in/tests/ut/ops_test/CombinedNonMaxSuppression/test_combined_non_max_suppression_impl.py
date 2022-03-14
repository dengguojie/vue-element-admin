#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
CombinedNonMaxSuppression ut test
"""
from unittest.mock import MagicMock
from unittest.mock import patch
from op_test_frame.ut import OpUT

ut_case = OpUT("CombinedNonMaxSuppression",
               "impl.combined_non_max_suppression",
               "combined_non_max_suppression")


def get_dict(_shape, dtype="float16"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": "ND", "ori_shape": _shape, "ori_format": "ND"}


def get_dict_const(value, dtype="float16"):
    return {"shape": [1], "dtype": dtype, "format": "ND", "ori_shape": [1],
            "ori_format": "ND", "const_value": (value,)}


# pylint: disable=unused-argument,missing-docstring
def get_impl_list(batch_size, num_boxes, num_class, num_class_boxes,
                  score_threshold, iou_threshold, max_size_per_class,
                  max_total_size, change_coordinate_frame, is_need_clip, is_need_valid):
    if num_class_boxes == 0:
        boxes_shape = [batch_size, 4, num_boxes]
    else:
        boxes_shape = [batch_size, num_class_boxes, 4, num_boxes]
    if num_class == 0:
        score_shape = [batch_size, num_boxes]
    else:
        score_shape = [batch_size, num_class, num_boxes]
    if is_need_clip:
        clip_shape = [batch_size, 4]
    else:
        clip_shape = None
    if is_need_valid:
        num_shape = [batch_size]
    else:
        num_shape = None

    output_boxes_shape = [batch_size, 4, max_total_size]
    output_score_shape = [batch_size, max_total_size]
    output_valid_shape = [batch_size, 8]
    input_list = [get_dict(boxes_shape), get_dict(score_shape)]
    const_list = [get_dict_const(max_size_per_class, "int32"),
                  get_dict_const(max_total_size, "int32"),
                  get_dict_const(iou_threshold, "float32"),
                  get_dict_const(score_threshold, "float32")
                  ]
    input_list = input_list + const_list
    output_list = [get_dict(output_boxes_shape), get_dict(output_score_shape), get_dict(output_score_shape),
                   get_dict(output_valid_shape, "int32")]
    par_list = [True, False]

    return input_list + output_list + par_list


case1 = {"params": get_impl_list(1, 29782, 1, 1, 0.5, 0.5, 100, 100, False, True, False),
         "case_name": "faster_rcnn_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": get_impl_list(8, 100, 90, 90, 0.5, 0.5, 100, 100, True, True, True),
         "case_name": "faster_rcnn_case_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": get_impl_list(4, 51150, 6, 1, 0.5, 0.5, 200, 300, False, True, False),
         "case_name": "ssd_mobile_1",
         "expect": "success",
         "support_expect": True}

case4 = {"params": get_impl_list(2, 190000, 0, 0, 0.5, 0.5, 200, 300, False, False, False),
         "case_name": "rainet",
         "expect": "success",
         "support_expect": True}

case5 = {"params": get_impl_list(2, 1000, 2, 2, 0.5, 0.5, 5, 100, False, False, False),
         "case_name": "one_less",
         "expect": "success",
         "support_expect": True}

case6 = {"params": get_impl_list(2, 1000, 150, 150, 0.5, 0.5, 300, 100, False, False, False),
         "case_name": "one_more_out",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [get_dict([1, 4, 200])] + get_impl_list(1, 200, 150, 150, 0.5, 0.5,
                                                           100, 100, False, False, False)[1:],
         "case_name": "check_shape_failed",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": get_impl_list(2, 1000, 150, 150, 0.5, 0.5, 300, 0, False, False, False),
         "case_name": "check_max_total_size_failed",
         "expect": RuntimeError,
         "support_expect": True}

case9 = {"params": get_impl_list(51, 2, 150, 150, 0.5, 0.5, 100, 100, False, False, False),
         "case_name": "large_batches",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend920A", "Ascend310"], case1)
ut_case.add_case(["Ascend920A", "Ascend310"], case2)
ut_case.add_case(["Ascend920A", "Ascend310"], case3)
ut_case.add_case(["Ascend920A", "Ascend310"], case4)
ut_case.add_case(["Ascend920A", "Ascend310"], case5)
ut_case.add_case(["Ascend920A", "Ascend310"], case6)
ut_case.add_case(["Ascend920A", "Ascend310"], case7)
ut_case.add_case(["Ascend920A", "Ascend310"], case8)
ut_case.add_case(["Ascend920A", "Ascend310"], case9)

from impl.combined_non_max_suppression import check_supported


# 'pylint: disable=unused-argument,unused-variable
def test_check_support(test_arg):
    res = check_supported({'shape': [1, 1, 4, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [1, 1, 4, 29782],
                           'ori_format': 'ND'},
                          {'shape': [1, 1, 29782], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [1, 1, 29782],
                           'ori_format': 'ND'},
                          {'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (100,)},
                          {'shape': [1], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (100,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)},
                          {'shape': [1], 'dtype': 'float32', 'format': 'ND', 'ori_shape': [1], 'ori_format': 'ND',
                           'const_value': (0.5,)},
                          {'shape': [1, 4, 100], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [1, 4, 100],
                           'ori_format': 'ND'},
                          {'shape': [1, 100], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [1, 100],
                           'ori_format': 'ND'},
                          {'shape': [1, 100], 'dtype': 'float16', 'format': 'ND', 'ori_shape': [1, 100],
                           'ori_format': 'ND'},
                          {'shape': [1, 8], 'dtype': 'int32', 'format': 'ND', 'ori_shape': [1, 8], 'ori_format': 'ND'},
                          True, False)


ut_case.add_cust_test_func(test_func=test_check_support)

ut_case.run(['Ascend920A', 'Ascend310'])
