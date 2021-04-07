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

L1LossGrad ut case
"""
import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("L1LossGrad", "impl.dynamic.l1_loss_grad", "l1_loss_grad")


case1 = {"params": [{"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #predict
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #grads
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #label
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, 
                    "none"
                    ],
         "case_name": "L1LossGrad_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #predict
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #grads
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #label
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, 
                    "sum"
                    ],
         "case_name": "L1LossGrad_dynamic_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #predict
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #grads
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #label
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, 
                    "mean"
                    ],
         "case_name": "L1LossGrad_dynamic_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #predict
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #grads
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #label
                    {"shape":(-1,-1,4), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,4), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, 
                    "mean"
                    ],
         "case_name": "L1LossGrad_dynamic_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape":(-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #predict
                    {"shape":(-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #grads
                    {"shape":(-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, #label
                    {"shape":(-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(4,4)]}, 
                    "mean"
                    ],
         "case_name": "L1LossGrad_dynamic_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape":(-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(4,4)]}, #predict
                    {"shape":(-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(4,4)]}, #grads
                    {"shape":(-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(4,4)]}, #label
                    {"shape":(-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(4,4)]}, 
                    "mean"
                    ],
         "case_name": "L1LossGrad_dynamic_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape":(-1,-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(1,None),(4,4)]}, #predict
                    {"shape":(-1,-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(1,None),(4,4)]}, #grads
                    {"shape":(-1,-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(1,None),(4,4)]}, #label
                    {"shape":(-1,-1,-1,-1,-1), "dtype":"float16", "format":"ND", "ori_shape":(-1,-1,-1,-1,-1), "ori_format":"ND", "range":[(1,None),(1,None),(1,None),(1,None),(4,4)]}, 
                    "mean"
                    ],
         "case_name": "L1LossGrad_dynamic_7",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case7)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)