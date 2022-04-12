#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import numpy as np

ut_case = BroadcastOpUT("fill_v2_d", "impl.fill_v2_d", "fill_v2_d")

def calc_expect_func(output_z, value, shape):
    res = np.ones(shape) * value
    return [res, ]

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, ), "shape": (32, ),
    "param_type": "output"}, 16.0, (32, )],
    "calc_expect_func": calc_expect_func
})