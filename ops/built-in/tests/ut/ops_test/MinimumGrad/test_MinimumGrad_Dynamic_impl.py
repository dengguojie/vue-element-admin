#!/usr/bin/env python
# -*- coding :UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("MinimumGrad","impl.dynamic.minimum_grad","minimum_grad")
case1={
    "params":[
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
    ],
    "case_name":"MinimumGrad_1",
    "expect":"success",
    "support_expect":True
}

ut_case.add_case("Ascend910A",case1)

if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")