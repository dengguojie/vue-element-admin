#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
test for SmoothL1Loss
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("SmoothL1Loss", "impl.dynamic.smooth_l1_loss", "smooth_l1_loss")

case1 = {
    "params": [{
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
    }, 1.0],
    "case_name": "SmoothL1Loss_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100)]
    }, 1.0],
    "case_name": "SmoothL1Loss_2",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
    ut_case.run("Ascend310")
