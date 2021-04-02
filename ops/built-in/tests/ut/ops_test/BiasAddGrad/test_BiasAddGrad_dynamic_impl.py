#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BiasAddGrad", "impl.dynamic.bias_add_grad", "bias_add_grad")

case1 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "NHWC", "ori_shape": (2,3),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100)]}, #x
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NHWC", "ori_shape":(2,3),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100)]},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_1",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)

if __name__ == '__main__':
    # ut_case.run(["Ascend310"])
    # ut_case.run(["Ascend710"])
    ut_case.run(["Ascend910A"])
