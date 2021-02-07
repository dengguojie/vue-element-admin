#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("KLDiv", "impl.dynamic.kl_div", "kl_div")

case1 = {
  "params": [
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 1),]},
    "batchmean"
  ],
  "case_name": "kl_div_dynamic_1",
  "expect": "success",
  "format_expect": [],
  "support_expect": True
}

ut_case.add_case(["Ascend910"], case1)

with te.op.dynamic():
    ut_case.run("Ascend910")

