"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

BoundingBoxDecode ut case
"""
import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("SmoothL1LossGradV2", "impl.dynamic.smooth_l1_loss_grad_v2", "smooth_l1_loss_grad_v2")

case1 = {
        "params": 
            [{
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, 1.0, "mean"],
        "case_name": "SmoothL1LossGradV2_1",
        "expect": "success",
        "support_expect": True}

case2 = {
        "params": 
            [{
                "shape": (-1, -1),
                "dtype": "int32",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, 1.0, "mean"],
        "case_name": "SmoothL1LossGradV2_2",
        "expect": RuntimeError,
        "support_expect": True}

case3 = {
        "params": 
            [{
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "int32",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, 1.0, "mean"],
        "case_name": "SmoothL1LossGradV2_3",
        "expect": RuntimeError,
        "support_expect": True}

case4 = {
        "params":
            [{
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "int32",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, 1.0,"mean"],
        "case_name": "SmoothL1LossGradV2_4",
        "expect": RuntimeError,
        "support_expect": True}

case5 = {
        "params": 
            [{
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, {
                "shape": (-1, -1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 4),
                "ori_format": "ND",
                "range": [(1, 100), (1, 100)]
            }, 1.0, "means"],
        "case_name": "SmoothL1LossGradV2_5",
        "expect": RuntimeError,
        "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)


if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
    exit(0)
