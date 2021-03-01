#!/usr/bin/env python
# -*- coding :UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("Expm1", "impl.dynamic.expm1", "expm1")

case1 = {
    "params":[
        {"shape":(-1,), "ori_shape":(2,4), "dtype":"float16", "format":"ND", "ori_format":"ND", "range":((1,100),)},
        {"shape":(-1,), "ori_shape":(2,4), "dtype":"float16", "format":"ND", "ori_format":"ND", "range":((1,100),)},
    ],
    "case_name":"Expm1_1",
    "expect":"success",
    "support_expect":True
}

ut_case.add_case(["Ascend910A", "Ascend710", "Ascend610"], case1)

if __name__=='__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")