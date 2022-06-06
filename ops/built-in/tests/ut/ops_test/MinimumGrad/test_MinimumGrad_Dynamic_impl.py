#!/usr/bin/env python
# -*- coding :UTF-8 -*-
from asyncio import FastChildWatcher
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

def test_check_support(test_arg):
    from impl.dynamic.minimum_grad import check_supported
    res = check_supported(        
                    {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(-1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    True,
                    True,
                    "dynamic_maximum_grad_check_support_case_01")
    assert res

    res = check_supported(        
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    True,
                    True,
                    "dynamic_maximum_grad_check_support_case_02")
    assert res == False

    res = check_supported(        
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    True,
                    True,
                    "dynamic_maximum_grad_check_support_case_03")
    assert res == False

    res = check_supported(        
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(2,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    {"shape":(1,),"dtype":"float32","format":"ND","ori_shape":(1,),"ori_format":"ND","range":[(1,10)]},
                    True,
                    False,
                    "dynamic_maximum_grad_check_support_case_04")
    assert res == True

ut_case.add_cust_test_func(test_func=test_check_support)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
