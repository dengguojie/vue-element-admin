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

InplaceIndexAdd ut case
"""

from op_test_frame.ut import OpUT
ut_case = OpUT("InplaceIndexAdd", "impl.dynamic.inplace_index_add", "inplace_index_add")

case1 = {"params": [{"shape": (-1, 8, 375), "dtype": "int32", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "range": [(1, 8)]},
                    {"shape": (-1, 3, 375), "dtype": "int32", "format": "ND", "ori_shape": (16, 3, 375), "ori_format": "ND", "range": [(15, 16), (3, 3), (375, 375)]},
                    {"shape": (16, 8, 375), "dtype": "int32", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(16, 16), (8, 8), (375, 375)]},
                    1,
                    ],
         "case_name": "InplaceIndexAdd_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, 32, 32, 375), "dtype": "int32", "format": "ND", "ori_shape": (64, 32, 32, 375), "ori_format": "ND", "range": [(63, 64), (32, 32), (32, 32), (375, 375)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (16,), "ori_format": "ND", "range": [(1, 16)]},
                    {"shape": (-1, 16, 32, 375), "dtype": "int32", "format": "ND", "ori_shape": (64, 16, 32, 375), "ori_format": "ND", "range": [(63, 64), (16, 16), (32, 32), (375, 375)]},
                    {"shape": (64, 32, 32, 375), "dtype": "int32", "format": "ND", "ori_shape": (64, 32, 32, 375), "ori_format": "ND", "range": [(63, 64), (32, 32), (32, 32), (375, 375)]},
                    1,
                    ],
         "case_name": "InplaceIndexAdd_dynamic_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 256, 256, 256, 24), "dtype": "int32", "format": "ND", "ori_shape": (64, 256, 256, 256, 24), "ori_format": "ND", "range": [(63, 64), (256, 256), (256, 256), (256, 256), (24, 24)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (32,), "ori_format": "ND", "range": [(1, 32)]},
                    {"shape": (-1, 32, 256, 256, 24), "dtype": "int32", "format": "ND", "ori_shape": (64, 32, 256, 256, 24), "ori_format": "ND", "range": [(63, 64), (32, 32), (256, 256), (256, 256), (24, 24)]},
                    {"shape": (64, 256, 256, 256, 24), "dtype": "int32", "format": "ND", "ori_shape": (64, 256, 256, 256, 24), "ori_format": "ND", "range": [(63, 64), (256, 256), (256, 256), (256, 256), (24, 24)]},
                    1,
                    ],
         "case_name": "InplaceIndexAdd_dynamic_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-1, 8, 375), "dtype": "int8", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "range": [(1, 8)]},
                    {"shape": (-1, 3, 375), "dtype": "int8", "format": "ND", "ori_shape": (16, 3, 375), "ori_format": "ND", "range": [(15, 16), (3, 3), (375, 375)]},
                    {"shape": (16, 8, 375), "dtype": "int8", "format": "ND", "ori_shape": (16, 8, 375), "ori_format": "ND", "range": [(16, 16), (8, 8), (375, 375)]},
                    1,
                    ],
         "case_name": "InplaceIndexAdd_dynamic_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
