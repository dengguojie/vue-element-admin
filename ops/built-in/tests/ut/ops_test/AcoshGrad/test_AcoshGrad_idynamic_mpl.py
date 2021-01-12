#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("AcoshGrad", "impl.dynamic.acosh_grad", "acosh_grad")

case1 = {
    "params": [
        {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(15, 32),"ori_format":"ND", "range":[(1, 100)]},
        {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(15, 32),"ori_format":"ND", "range":[(1, 100)]},
        {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(15, 32),"ori_format":"ND", "range":[(1, 100)]}
    ],
    "case_name": "AcoshGrad_dynamic_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {"shape":(-1, -1, -1), "dtype":"float16", "format":"ND", "ori_shape":(2, 4, 4),"ori_format":"ND", "range":[(1, 10)]},
        {"shape":(-1, -1, -1), "dtype":"float16", "format":"ND", "ori_shape":(2, 4, 4),"ori_format":"ND", "range":[(1, 10)]},
        {"shape":(-1, -1, -1), "dtype":"float16", "format":"ND", "ori_shape":(2, 4, 4),"ori_format":"ND", "range":[(1, 10)]}
    ],
    "case_name": "AcoshGrad_dynamic_2",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910", "Ascend610", "Ascend710"], case1)
ut_case.add_case(["Ascend910", "Ascend610", "Ascend710"], case2)

with te.op.dynamic():
    ut_case.run(["Ascend910", "Ascend610", "Ascend710"])
