"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

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
from op_test_frame.ut import OpUT
ut_case = OpUT("MseLossGrad", None, None)


case1 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape":(1,),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    "mean"
                    ],
         "case_name": "mse_loss_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    "sum"
                    ],
         "case_name": "mse_loss_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    "none"
                    ],
         "case_name": "mse_loss_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)


if __name__ == '__main__':
    ut_case.run()
