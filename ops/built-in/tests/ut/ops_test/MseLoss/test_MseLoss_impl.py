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

MseLoss ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("MseLoss", None, None)

case1 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape":(1,),"ori_format": "ND"},
                    "mean"
                    ],
         "case_name": "MseLoss_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape":(1,),"ori_format": "ND"},
                    "sum"
                    ],
         "case_name": "MseLoss_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    "none"
                    ],
         "case_name": "MseLoss_3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case3)


if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend310P3"])
