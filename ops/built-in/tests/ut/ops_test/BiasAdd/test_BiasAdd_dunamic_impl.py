#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te

from op_test_frame.ut import OpUT
ut_case = OpUT("BiasAdd", "impl.dynamic.bias_add", "bias_add")

case1 = {"params": [{"shape": (-1,-1,4), "dtype": "float16", "format": "NHWC", "ori_shape": (-1,-1,4),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NHWC", "ori_shape": (4,),"ori_format": "NHWC","range":[(1, 100)]},
                    {"shape": (-1,-1,4), "dtype": "float16", "format": "NHWC", "ori_shape":(-1,-1,4),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100)]},
                    ],
         "case_name": "BiasAdd_1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)

