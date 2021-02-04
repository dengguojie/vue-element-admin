# !/user/bin/env python
# -*- coding: utf-8 -*-
import te
from op_test_frame.ut import OpUT


ut_case = OpUT("OnesLike","impl.dynamic.ones_like","ones_like")

case1 = {"params":[
        {"shape": (-1,), "dtype": "float32", "format": "ND","ori_shape": (-1,),
        "ori_format": "ND", "range": [(1,100)]},
        {"shape": (-1,), "dtype": "float32", "format": "ND",
        "ori_shape": (-1,), "ori_format": "ND", "range": [(1,100)]},],
        "except": "success",
        "format_except": [],
        "support_except": True}

ut_case.add_case(["Ascend910A"], case1)


with te.op.dynamic():
    ut_case.run("Ascend910A")
