#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("BNInferenceD", "impl.dynamic.bninference_d", "bninference_d")

case1 = {
    "params": [
        {"shape": (-1,-1,-1,-1),"dtype":"float16","format":"NCHW","ori_shape":(1,16,10,10),"ori_format":"NCHW","range":[(1,None),
        (16,16),(10,10),(10,10)]},
        {"shape": (-1,),"dtype":"float16","format":"ND","ori_shape":(16,),"ori_format":"ND","range":[(1,None)]},
        {"shape": (-1,),"dtype": "float16","format":"ND","ori_shape":(16,),"ori_format":"ND","range":[(1,None)]},
        None,
        None,
        {"shape":(-1,-1,-1,-1),"dtype":"float16", "format":"NCHW", "ori_shape":(1,16,10,10),"ori_format":"NCHW","range":[(1,None),
        (1,None),(1,None),(1,None)]},
        0.999,
        0.001,
        True,
        1
    ],
    "case_name": "bninferenced_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [
        {"shape": (-1,-1,-1,-1),"dtype":"float16","format":"NCHW","ori_shape":(1,16,10,10),"ori_format":"NCHW","range":[(1,None), (1,None),(1,None),(1,None)]},
        {"shape": (-1,),"dtype":"float16","format":"ND","ori_shape":(16,),"ori_format":"ND","range":[(1,None)]},
        {"shape": (-1,),"dtype": "float16","format":"ND","ori_shape":(16,),"ori_format":"ND","range":[(1,None)]},
        {"shape": (-1,),"dtype":"float16","format":"ND","ori_shape":(16,),"ori_format":"ND","range":[(1,None)]},
        {"shape": (-1,),"dtype":"float16","format":"ND","ori_shape":(16,),"ori_format":"ND","range":[(1,None)]},
        {"shape":(-1,-1,-1,-1),"dtype":"float16", "format":"NCHW", "ori_shape":(1,16,10,10),"ori_format":"NCHW","range":[(1,None), (1,None),(1,None),(1,None)]},
        0.999,
        0.001,
        True,
        1
    ],
    "case_name": "bninferenced_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(["Ascend910A"])
