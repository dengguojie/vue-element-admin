#!/usr/bin/env python
# -*- coding :UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("MaximumGrad","impl.dynamic.maximum_grad","maximum_grad")
case1={
    "params":[
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
        {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
    ],
    "case_name":"MaximumGrad_1",
    "expect":"success",
    "support_expect":True
}

ut_case.add_case("Ascend910",case1)

with te.op.dynamic():
    ut_case.run("Ascend910")