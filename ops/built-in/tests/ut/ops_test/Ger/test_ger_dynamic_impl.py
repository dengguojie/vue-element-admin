"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Ger ut case
"""
import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("Ger", "impl.dynamic.ger", "ger")


case1 = {"params": [{"shape":(-1,),   "dtype":"float16", "format":"ND", "ori_shape":(-1,),   "ori_format":"ND", "range":[(1,None)]}, #x1
                    {"shape":(-1,),   "dtype":"float16", "format":"ND", "ori_shape":(-1,),   "ori_format":"ND", "range":[(1,None)]}, #x2
                    {"shape":(-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1), "ori_format":"ND", "range":[(1,None),(1,None)]}],
         "case_name": "Ger_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape":(-1,),    "dtype":"float16", "format":"NCHW", "ori_shape":(-1,),    "ori_format":"NCHW", "range":[(1,None)]},  #x1
                    {"shape":(200,),   "dtype":"float16", "format":"NCHW", "ori_shape":(200,),   "ori_format":"NCHW", "range":[(200,200)]}, #x2
                    {"shape":(-1,200), "dtype":"float16", "format":"NCHW", "ori_shape":(-1,200), "ori_format":"NCHW", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape":(-1,),   "dtype":"float32", "format":"ND", "ori_shape":(-1,),   "ori_format":"ND", "range":[(1,None)]}, #x1
                    {"shape":(-1,),   "dtype":"float32", "format":"ND", "ori_shape":(-1,),   "ori_format":"ND", "range":[(1,None)]}, #x2
                    {"shape":(-1,-1), "dtype":"float32", "format":"ND", "ori_shape":(-1,-1), "ori_format":"ND", "range":[(1,None),(1,None)]}],
         "case_name": "Ger_dynamic_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape":(-1,),    "dtype":"float32", "format":"NCHW", "ori_shape":(-1,),    "ori_format":"NCHW", "range":[(1,None)]},  #x1
                    {"shape":(200,),   "dtype":"float32", "format":"NCHW", "ori_shape":(200,),   "ori_format":"NCHW", "range":[(200,200)]}, #x2
                    {"shape":(-1,200), "dtype":"float32", "format":"NCHW", "ori_shape":(-1,200), "ori_format":"NCHW", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape":(1,),   "dtype":"float32", "format":"NCHW", "ori_shape":(1,),   "ori_format":"NCHW", "range":[(1,1)]},  #x1
                    {"shape":(-1,),  "dtype":"float32", "format":"NCHW", "ori_shape":(-1,),  "ori_format":"NCHW", "range":[(1,None)]}, #x2
                    {"shape":(1,-1), "dtype":"float32", "format":"NCHW", "ori_shape":(1,-1), "ori_format":"NCHW", "range":[(1,1),(1,None)]}],
         "case_name": "Ger_dynamic_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape":(-1,),    "dtype":"float32", "format":"NHWC", "ori_shape":(-1,),    "ori_format":"NHWC", "range":[(1,None)]},  #x1
                    {"shape":(200,),   "dtype":"float32", "format":"NHWC", "ori_shape":(200,),   "ori_format":"NHWC", "range":[(200,200)]}, #x2
                    {"shape":(-1,200), "dtype":"float32", "format":"NHWC", "ori_shape":(-1,200), "ori_format":"NHWC", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape":(-1,),    "dtype":"float32", "format":"NC1HWC0", "ori_shape":(-1,),    "ori_format":"NC1HWC0", "range":[(1,None)]},  #x1
                    {"shape":(200,),   "dtype":"float32", "format":"NC1HWC0", "ori_shape":(200,),   "ori_format":"NC1HWC0", "range":[(200,200)]}, #x2
                    {"shape":(-1,200), "dtype":"float32", "format":"NC1HWC0", "ori_shape":(-1,200), "ori_format":"NC1HWC0", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape":(-1,),    "dtype":"float32", "format":"ND", "ori_shape":(-1,),    "ori_format":"ND", "range":[(1,None)]},  #x1
                    {"shape":(200,),   "dtype":"float32", "format":"ND", "ori_shape":(200,),   "ori_format":"ND", "range":[(200,200)]}, #x2
                    {"shape":(-1,200), "dtype":"float32", "format":"ND", "ori_shape":(-1,200), "ori_format":"ND", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape":(-1,),    "dtype":"int32", "format":"ND", "ori_shape":(-1,),    "ori_format":"ND", "range":[(1,None)]},  #x1
                    {"shape":(200,),   "dtype":"int32", "format":"ND", "ori_shape":(200,),   "ori_format":"ND", "range":[(200,200)]}, #x2
                    {"shape":(-1,200), "dtype":"int32", "format":"ND", "ori_shape":(-1,200), "ori_format":"ND", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_9",
         "expect": RuntimeError,
         "support_expect": True}

case10 = {"params": [{"shape":(-1,),    "dtype":"float32", "format":"ND", "ori_shape":(-1,),    "ori_format":"ND", "range":[(1,None)]},  #x1
                    {"shape":(200,),   "dtype":"float16", "format":"ND", "ori_shape":(200,),   "ori_format":"ND", "range":[(200,200)]}, #x2
                    {"shape":(-1,200), "dtype":"float32", "format":"ND", "ori_shape":(-1,200), "ori_format":"ND", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_10",
         "expect": RuntimeError,
         "support_expect": True}

case11 = {"params": [{"shape":(-1,200), "dtype":"float32", "format":"ND", "ori_shape":(-1,200), "ori_format":"ND", "range":[(1,None),(200,200)]},  #x1
                     {"shape":(-1,200), "dtype":"float32", "format":"ND", "ori_shape":(-1,200), "ori_format":"ND", "range":[(1,None),(200,200)]},  #x2
                     {"shape":(-1,200), "dtype":"float32", "format":"ND", "ori_shape":(-1,200), "ori_format":"ND", "range":[(1,None),(200,200)]}],
         "case_name": "Ger_dynamic_11",
         "expect": RuntimeError,
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case4)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case5)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case6)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case7)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case8)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case9)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case10)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case11)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710", "Ascend310"])
    exit(0)