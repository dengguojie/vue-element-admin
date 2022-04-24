#!/usr/bin/env python
# -*- coding :UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("BNInfer","impl.dynamic.bn_infer","bn_infer")
case1={
    "params":[
        {"shape":(-1,1,1,1,16),"dtype":"float32","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float32","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float32","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float32","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float32","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float32","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        0.0001    
    ],
    "case_name":"BNInfer_1",
    "expect":"success",
    "support_expect":True
}
case2={
    "params":[
        {"shape":(-1,1,1,1,16),"dtype":"float16","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float16","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float16","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float16","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float16","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        {"shape":(-1,1,1,1,16),"dtype":"float16","format":"NC1HWC0","ori_shape":(1,1,1,1,16),"ori_format":"NC1HWC0","range":[(1,10),(1,1),(1,1),(1,1),(16,16)]},
        0.0001    
    ],
    "case_name":"BNInfer_1",
    "expect":"success",
    "support_expect":True
}

ut_case.add_case("Ascend910A",case1)
ut_case.add_case("Ascend910A",case2)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
