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

HardtanhGrad ut case
"""

from op_test_frame.ut import OpUT
ut_case = OpUT("HardtanhGrad", "impl.dynamic.hardtanh_grad", "hardtanh_grad")

case1 = {"params": [{"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(16, 16), (8, 8), (375, 375)]},
                    -1.0,
                    1.0
                    ],
         "case_name": "HardtanhGrad_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(16, 16), (8, 8), (375, 375)]},
                    0.0,
                    1.0
                    ],
         "case_name": "HardtanhGrad_dynamic_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(16, 16), (8, 8), (375, 375)]},
                    -1.0,
                    0.0
                    ],
         "case_name": "HardtanhGrad_dynamic_3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
