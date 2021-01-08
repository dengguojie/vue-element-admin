#!/usr/bin/env python
# -*- coding :UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Floor","impl.dynamic.floor","floor")

case1 = {
    "params":[
        {"shape":(-1,),"ori_shape":(2,4),"dtype":"float16","format":"ND","ori_format":"ND","range":((1,100),)},
        {"shape":(-1,),"ori_shape":(2,4),"dtype":"float16","format":"ND","ori_format":"ND","range":((1,100),)},
    ],
    "case_name":"Floor_1",
    "expect":"success",
    "support_expect":True
}

ut_case.add_case(["Ascend910", "Ascend710", "Ascend310"], case1)

if __name__=='__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")