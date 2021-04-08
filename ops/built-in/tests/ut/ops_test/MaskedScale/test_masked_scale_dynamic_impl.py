#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
masked_scale
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("MaskedScale", "impl.dynamic.masked_scale", "masked_scale")

case1 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND","range":[(1, 100)]}, 
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND","range":[(1, 100)]}, 
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),
                     "ori_format": "ND","range":[(1, 100)]},
                    1.0],
         "case_name": "MaskedScale_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
                     "ori_format": "ND","range":[(1,None),(4,4)]}, 
                    {"shape": (-1, -1), "dtype": "int8", "format": "ND", "ori_shape": (-1, -1),
                     "ori_format": "ND","range":[(1,None),(4,4)]}, 
                    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
                     "ori_format": "ND","range":[(1,None),(4,4)]},
                    1.0],
         "case_name": "MaskedScale_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 10),
                     "ori_format": "NHWC","range":[(1,None),(10, 10)]}, 
                    {"shape": (-1, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 10),
                     "ori_format": "NHWC","range":[(1,None),(10, 10)]}, 
                    {"shape": (-1, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 10),
                     "ori_format": "NHWC","range":[(1,None),(10, 10)]},
                    1.0],
         "case_name": "MaskedScale_3",
         "expect": "success",
         "support_expect": True}
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])