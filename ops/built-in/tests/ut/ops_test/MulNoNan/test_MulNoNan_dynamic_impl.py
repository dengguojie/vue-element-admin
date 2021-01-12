#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("MulNoNan", "impl.dynamic.mul_no_nan", "mul_no_nan")

case1 = {
    "params": [
        {"shape":(-1,), "dtype":"float32", "format":"NHWC", "ori_shape":(2, ),"ori_format":"NHWC", "range":[(1, None)]},
        {"shape":(1,), "dtype":"float32", "format":"NHWC", "ori_shape":(1, ),"ori_format":"NHWC", "range":[(1, 1)]},
        {"shape":(-1,), "dtype":"float32", "format":"NHWC", "ori_shape":(2, ),"ori_format":"NHWC", "range":[(1, None)]}
    ],
    "case_name": "MulNoNan_dynamic_1",
    "expect": "success",
    "format_expect":[],
    "support_expect": True
}

ut_case.add_case(["Ascend910", "Ascend610", "Ascend710"], case1)

with te.op.dynamic():
    ut_case.run(["Ascend910", "Ascend610", "Ascend710"])
