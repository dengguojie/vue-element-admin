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

MseLossGrad ut case
"""
import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("MseLossGrad", "impl.dynamic.mse_loss_grad", "mse_loss_grad")

case1 = {"params": [{"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND","range":[(15,16),(8,8),(375,375)]},
                    {"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND","range":[(15,16),(8,8),(375,375)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape":(1,),"ori_format": "ND","range":[(1,1)]},
                    {"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND","range":[(15,16),(8,8),(375,375)]},
                    "mean"
                    ],
         "case_name": "MseLossGrad_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, 8, 36), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 36),"ori_format": "ND","range":[(15,16),(8,8),(36,36)]},
                    {"shape": (-1, 8, 36), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 36),"ori_format": "ND","range":[(15,16),(8,8),(36,36)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape":(1,),"ori_format": "ND","range":[(1,1)]},
                    {"shape": (-1, 8, 36), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 36),"ori_format": "ND","range":[(15,16),(8,8),(36,36)]},
                    "sum"
                    ],
         "case_name": "MseLossGrad_dynamic_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 8, 36), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 36),"ori_format": "ND","range":[(15,16),(8,8),(36,36)]},
                    {"shape": (-1, 8, 36), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 36),"ori_format": "ND","range":[(15,16),(8,8),(36,36)]},
                    {"shape": (-1, 8, 36), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 36),"ori_format": "ND","range":[(15,16),(8,8),(36,36)]},
                    {"shape": (-1, 8, 36), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 36),"ori_format": "ND","range":[(15,16),(8,8),(36,36)]},
                    "none"
                    ],
         "case_name": "MseLossGrad_dynamic_3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)


if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(["Ascend910A"])
